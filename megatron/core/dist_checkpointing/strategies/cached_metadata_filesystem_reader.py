# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

""" FS Reader with metadata cached support. """

import io
import os
import logging
from typing import Any, Dict, Union, cast

import torch
from torch import Tensor
from torch.distributed.checkpoint import FileSystemReader, Metadata, ReadItem
from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner
from torch.distributed.checkpoint.planner import LoadItemType
from torch.futures import Future
from torch.distributed._shard._utils import narrow_tensor_by_index

logger = logging.getLogger(__name__)


class CachedMetadataFileSystemReader(FileSystemReader):
    """
    Extends FileSystemReader to cache metadata for improved performance.

    Metadata is shared across all reader instances that use the same checkpoint
    directory (same path), since the loaded metadata is identical.

    Attributes:
        _metadata_cache (Dict[str, Metadata]): Class-level cache keyed by checkpoint path.
    """

    _metadata_cache: Dict[str, Metadata] = {}

    def __init__(self, path: Union[str, os.PathLike]) -> None:
        """
        Initialize with file system path.

        Args:
            path (Union[str, os.PathLike]): Path to the checkpoint directory or file.
        """
        super().__init__(path=path)
        self._cache_key = os.path.abspath(os.fspath(path))

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        # group requests by file
        per_file: dict[str, list[ReadItem]] = {}
        for read_item in plan.items:
            item_md: Any = self.storage_data[read_item.storage_index]
            path = item_md.relative_path
            per_file.setdefault(path, []).append(read_item)

        for relative_path, reqs in per_file.items():
            logger.info(f"Reading file {relative_path}")
            new_path = self.fs.concat_path(self.path, relative_path)
            with self.fs.create_stream(new_path, "rb") as stream:
                # TODO sort by offset and cache the reading
                for req in reqs:
                    item_md = self.storage_data[req.storage_index]
                    file_slice = self._slice_file(stream, item_md)
                    transform_from = self.transforms.transform_load_stream(
                        req,
                        # This field wasn't present in older
                        # implementations so provide a fallback.
                        item_md.transform_descriptors or (),
                        file_slice,
                    )

                    if req.type == LoadItemType.BYTE_IO:
                        read_bytes = io.BytesIO(transform_from.read(-1))
                        read_bytes.seek(0, io.SEEK_END)
                        byte_len = read_bytes.tell()
                        read_bytes.seek(0)
                        logger.info(f"Byte_IO, {req.storage_index.fqn}, {byte_len}, {relative_path}")
                        planner.load_bytes(req, read_bytes)
                    else:
                        if transform_from.seekable():
                            seekable = transform_from
                        else:
                            # torch.load requires a seekable input, so read the transform
                            # stream now and store the output if needed
                            seekable = io.BytesIO(transform_from.read(-1))
                            seekable.seek(0)

                        tensor = cast(
                            Tensor,
                            torch.load(
                                seekable,
                                map_location="cpu",
                                weights_only=True,
                            ),
                        )
                        tensor = narrow_tensor_by_index(
                            tensor, req.storage_offsets, req.lengths
                        )
                        target_tensor = planner.resolve_tensor(req).detach()

                        logger.info(f"Tensor, {req.storage_index.fqn},  {target_tensor.size()}, {relative_path}")
                        if target_tensor.size() != tensor.size():
                            raise AssertionError(
                                f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                            )
                        target_tensor.copy_(tensor)
                        planner.commit_tensor(req, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    def read_metadata(self) -> Metadata:
        """
        Read metadata from file system, caching for subsequent calls.
        Shared across instances when the checkpoint directory is the same.

        Returns:
            Metadata: Checkpoint metadata.
        """
        if self._cache_key not in CachedMetadataFileSystemReader._metadata_cache:
            CachedMetadataFileSystemReader._metadata_cache[
                self._cache_key
            ] = super().read_metadata()
        return CachedMetadataFileSystemReader._metadata_cache[self._cache_key]
