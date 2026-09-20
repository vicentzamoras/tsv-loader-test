"""TSVDataLoader: a standalone DashAI plugin.

Minimal, illustrative dataloader that ingests tab-separated (``.tsv``) files.
It is a thin variant of DashAI's built-in CSV loader: it delegates to
HuggingFace ``datasets`` with a fixed tab delimiter.
"""

from typing import TYPE_CHECKING, Any, Dict

from DashAI.back.core.schema_fields import enum_field, schema_field, string_field
from DashAI.back.core.schema_fields.base_schema import BaseSchema
from DashAI.back.core.utils import MultilingualString
from DashAI.back.dataloaders.classes.dataloader import BaseDataLoader

if TYPE_CHECKING:
    from DashAI.back.dataloaders.classes.dashai_dataset import DashAIDataset


class TSVDataLoaderSchema(BaseSchema):
    """Hyperparameters for :class:`TSVDataLoader`."""

    header: schema_field(
        string_field(),
        "infer",
        description=MultilingualString(
            en="Row number with the column labels, or 'infer' to detect them.",
            es="Número de fila con las etiquetas de columna, o 'infer' para detectar.",
        ),
        alias=MultilingualString(en="Header", es="Encabezado"),
    )  # type: ignore

    encoding: schema_field(
        enum_field(["utf-8", "latin1", "cp1252", "iso-8859-1"]),
        "utf-8",
        description=MultilingualString(
            en="Text encoding used to read the file.",
            es="Codificación de texto usada para leer el archivo.",
        ),
        alias=MultilingualString(en="Encoding", es="Codificación"),
    )  # type: ignore


class TSVDataLoader(BaseDataLoader):
    """Load tab-separated (``.tsv``) files into a DashAI dataset."""

    COMPATIBLE_COMPONENTS = ["TabularClassificationTask"]
    SCHEMA = TSVDataLoaderSchema
    SUPPORTED_EXTENSIONS = frozenset({".tsv", ".txt", ".zip"})

    DESCRIPTION = MultilingualString(
        en="Data loader for tab-separated (TSV) tabular files. UPDATE",
        es="Cargador de datos para archivos tabulares separados por tabulación (TSV).",
    )
    DISPLAY_NAME = MultilingualString(
        en="TSV Data Loader",
        es="Cargador de Datos TSV",
    )

    def _read_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Only forward the keys HuggingFace's csv builder understands.
        clean: Dict[str, Any] = {"delimiter": "\t"}
        if params.get("header") is not None:
            clean["header"] = params["header"]
        if params.get("encoding"):
            clean["encoding"] = params["encoding"]
        return clean

    def load_data(
        self,
        filepath_or_buffer: str,
        temp_path: str,
        params: Dict[str, Any],
        n_sample: int | None = None,
    ) -> "DashAIDataset":
        from DashAI.back.dataloaders.classes.dashai_dataset import to_dashai_dataset
        from datasets import load_dataset

        prepared_path, path_type = self.prepare_files(filepath_or_buffer, temp_path)
        data_arg = (
            {"data_files": prepared_path}
            if path_type == "file"
            else {"data_dir": prepared_path}
        )
        dataset = load_dataset(
            "csv",
            cache_dir=temp_path,
            **data_arg,
            **self._read_params(params),
        )
        return to_dashai_dataset(dataset)

    def load_preview(
        self,
        filepath_or_buffer: str,
        params: Dict[str, Any],
        n_rows: int = 100,
    ):
        from itertools import islice

        import pandas as pd
        from datasets import load_dataset

        dataset_stream = load_dataset(
            "csv",
            data_files=filepath_or_buffer,
            streaming=True,
            split="train",
            **self._read_params(params),
        )
        return pd.DataFrame(list(islice(dataset_stream, n_rows)))
