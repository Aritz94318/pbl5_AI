import pytest
import numpy as np
import pydicom
from io import BytesIO
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    generate_uid,
)

from service_api import load_dicom_as_pil_from_bytes


@pytest.fixture
def sample_dicom_bytes():
    # ✅ File Meta obligatorio para escribir en formato DICOM "enforce_file_format=True"
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()

    ds = Dataset()
    ds.file_meta = file_meta

    # ✅ SOP comúnmente requerido
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

    # --- Imagen mínima MONOCHROME2 ---
    ds.Rows = 32
    ds.Columns = 32
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0  # unsigned
    ds.PlanarConfiguration = 0  # no aplica realmente si SamplesPerPixel=1, pero no molesta

    rng = np.random.default_rng(seed=42)
    pixel_array = (rng.random((32, 32)) * 4095).astype(np.uint16)
    ds.PixelData = pixel_array.tobytes()

    buffer = BytesIO()

    # ✅ Sustituye write_like_original (deprecated) por enforce_file_format
    pydicom.dcmwrite(buffer, ds, enforce_file_format=True)
    return buffer.getvalue()


def test_invalid_dicom():
    with pytest.raises(Exception):
        load_dicom_as_pil_from_bytes(b"fake")


def test_valid_dicom(sample_dicom_bytes):
    img = load_dicom_as_pil_from_bytes(sample_dicom_bytes)
    assert img.mode == "L"
    assert img.size[0] > 10
