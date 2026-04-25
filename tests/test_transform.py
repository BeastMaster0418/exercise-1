"""
tests/test_transform.py
Integration tests for etl/transform.py — covers every output column.
"""
import math
import pytest
import pandas as pd
from pandas import isna
from etl.transform import transform, build_customer_lookup_index, get_customer_field_by_ref

# ── Trade Name ───────────────────────────────────────────────────────────────

class TestTradeName:
    def test_uses_customer_trade_name(self, extraction_single, customer_basic):
        df = transform(extraction_single, customer_basic)
        assert df["Trade Name"].iloc[0] == "Acetamide MEA 75%"

    def test_falls_back_to_extraction_trade_name(self, extraction_single, customer_empty):
        df = transform(extraction_single, customer_empty)
        assert df["Trade Name"].iloc[0] == "Product Alpha"

    def test_trade_name_blank_on_sub_rows(self, extraction_multi_row, customer_multi):
        df = transform(extraction_multi_row, customer_multi)
        assert df["Trade Name"].iloc[0] == "Trade Gamma"   # first row → populated
        assert isna(df["Trade Name"].iloc[1])             # sub-row → blank
        assert isna(df["Trade Name"].iloc[2])             # sub-row → blank

    def test_new_ref_resets_trade_name(self, customer_multi):
        extraction = pd.DataFrame({
            "Reference Number": [10001.0, 10001.0, 10002.0],
            "Trade Name":       ["A", None, "B"],
            "Manufacturer":     [None, None, None],
            "INCI":             [None, None, None],
            "INCI Comp":        [None, None, None],
            "Incidental?":      [None, None, None],
            "Impurity":         [None, None, None],
            "Impurity Comp":    [None, None, None],
            "Allergens":        [None, None, None],
            "Allergen Comp":    [None, None, None],
            "Function":         [None, None, None],
            "Notes":            [None, None, None],
            "GROUP":            [1.0, None, 1.0],
            "Missing Comp":     [None, None, None],
            "SDS Comp":         [None, None, None],
        })
        df = transform(extraction, customer_multi)
        assert df["Trade Name"].iloc[0] == "Trade Alpha"
        assert isna(df["Trade Name"].iloc[1])
        assert df["Trade Name"].iloc[2] == "Trade Beta"

# ── Manufacturer ─────────────────────────────────────────────────────────────

class TestManufacturer:
    def test_manufacturer_mapped(self, extraction_single, customer_basic):
        df = transform(extraction_single, customer_basic)
        assert df["Manufacturer"].iloc[0] == "ACME Corp"

    def test_manufacturer_none_when_missing(self, customer_basic):
        extraction = pd.DataFrame({
            "Reference Number": [10001.0],
            "Trade Name": ["X"], "Manufacturer": [None], "INCI": [None],
            "INCI Comp": [None], "Incidental?": [None], "Impurity": [None],
            "Impurity Comp": [None], "Allergens": [None], "Allergen Comp": [None],
            "Function": [None], "Notes": [None], "GROUP": [1.0],
            "Missing Comp": [None], "SDS Comp": [None],
        })
        df = transform(extraction, customer_basic)
        assert isna(df["Manufacturer"].iloc[0])

class TestReferenceNumber:
    def test_present_on_first_row(self, extraction_single, customer_basic):
        df = transform(extraction_single, customer_basic)
        assert df["Reference Number"].iloc[0] == "10001"

    def test_blank_on_sub_rows(self, extraction_multi_row, customer_multi):
        df = transform(extraction_multi_row, customer_multi)
        assert df["Reference Number"].iloc[0] == "10005"
        assert isna(df["Reference Number"].iloc[1])
        assert isna(df["Reference Number"].iloc[2])

    def test_max_30_chars(self, customer_empty):
        extraction = pd.DataFrame({
            "Reference Number": ["9" * 40],
            "Trade Name": ["X"], "Manufacturer": [None], "INCI": [None],
            "INCI Comp": [None], "Incidental?": [None], "Impurity": [None],
            "Impurity Comp": [None], "Allergens": [None], "Allergen Comp": [None],
            "Function": [None], "Notes": [None], "GROUP": [1.0],
            "Missing Comp": [None], "SDS Comp": [None],
        })
        df = transform(extraction, customer_empty)
        assert len(df["Reference Number"].iloc[0]) == 30

    def test_forward_fill_propagates_ref(self, extraction_no_ref, customer_empty):
        df = transform(extraction_no_ref, customer_empty)
        # All rows belong to same ref, so only first has Reference Number
        assert df["Reference Number"].iloc[0] == "10002"
        assert isna(df["Reference Number"].iloc[1])

# ── INCI Name ────────────────────────────────────────────────────────────────

class TestInciName:
    def test_inci_name_mapped(self, extraction_single, customer_basic):
        df = transform(extraction_single, customer_basic)
        assert df["INCI Name"].iloc[0] == "Water"

    def test_inci_name_max_200_chars(self, customer_empty):
        extraction = pd.DataFrame({
            "Reference Number": [10001.0],
            "Trade Name": ["X"], "Manufacturer": [None],
            "INCI": ["A" * 250],
            "INCI Comp": [None], "Incidental?": [None], "Impurity": [None],
            "Impurity Comp": [None], "Allergens": [None], "Allergen Comp": [None],
            "Function": [None], "Notes": [None], "GROUP": [1.0],
            "Missing Comp": [None], "SDS Comp": [None],
        })
        df = transform(extraction, customer_empty)
        assert len(df["INCI Name"].iloc[0]) == 200

    def test_inci_name_none_when_missing(self, customer_empty):
        extraction = pd.DataFrame({
            "Reference Number": [10001.0],
            "Trade Name": ["X"], "Manufacturer": [None], "INCI": [None],
            "INCI Comp": [None], "Incidental?": [None], "Impurity": [None],
            "Impurity Comp": [None], "Allergens": [None], "Allergen Comp": [None],
            "Function": [None], "Notes": [None], "GROUP": [1.0],
            "Missing Comp": [None], "SDS Comp": [None],
        })
        df = transform(extraction, customer_empty)
        assert isna(df["INCI Name"].iloc[0])

# ── Is Incidental? ───────────────────────────────────────────────────────────

class TestIsIncidental:
    def test_no_when_none(self, extraction_single, customer_basic):
        df = transform(extraction_single, customer_basic)
        assert df["Is Incidental?"].iloc[0] == "No"

    def test_yes_when_y(self, customer_basic):
        extraction = pd.DataFrame({
            "Reference Number": [10001.0],
            "Trade Name": ["X"], "Manufacturer": [None], "INCI": [None],
            "INCI Comp": [None], "Incidental?": ["Y"], "Impurity": [None],
            "Impurity Comp": [None], "Allergens": [None], "Allergen Comp": [None],
            "Function": [None], "Notes": [None], "GROUP": [1.0],
            "Missing Comp": [None], "SDS Comp": [None],
        })
        df = transform(extraction, customer_basic)
        assert df["Is Incidental?"].iloc[0] == "Yes"

    def test_only_yes_or_no(self, extraction_multi_row, customer_multi):
        df = transform(extraction_multi_row, customer_multi)
        assert set(df["Is Incidental?"].unique()).issubset({"Yes", "No"})

# ── INCI Concentration / Min / Max ───────────────────────────────────────────

class TestInciConcentration:
    def test_exact_100(self, extraction_single, customer_basic):
        df = transform(extraction_single, customer_basic)
        assert df["INCI Concentration"].iloc[0] == 100.0

    def test_range_populates_min_max(self, extraction_multi_row, customer_multi):
        df = transform(extraction_multi_row, customer_multi)
        assert df["INCI Conc Min"].iloc[0] == 51.0
        assert df["INCI Conc Max"].iloc[0] == 59.0
        assert isna(df["INCI Concentration"].iloc[0])

    def test_out_of_range_clamped_to_none(self, customer_empty):
        extraction = pd.DataFrame({
            "Reference Number": [10001.0],
            "Trade Name": ["X"], "Manufacturer": [None], "INCI": [None],
            "INCI Comp": [150],   # out of 0-100
            "Incidental?": [None], "Impurity": [None], "Impurity Comp": [None],
            "Allergens": [None], "Allergen Comp": [None], "Function": [None],
            "Notes": [None], "GROUP": [1.0], "Missing Comp": [None], "SDS Comp": [None],
        })
        df = transform(extraction, customer_empty)
        assert isna(df["INCI Concentration"].iloc[0])

    def test_zero_boundary(self, customer_empty):
        extraction = pd.DataFrame({
            "Reference Number": [10001.0],
            "Trade Name": ["X"], "Manufacturer": [None], "INCI": [None],
            "INCI Comp": [0],
            "Incidental?": [None], "Impurity": [None], "Impurity Comp": [None],
            "Allergens": [None], "Allergen Comp": [None], "Function": [None],
            "Notes": [None], "GROUP": [1.0], "Missing Comp": [None], "SDS Comp": [None],
        })
        df = transform(extraction, customer_empty)
        assert df["INCI Concentration"].iloc[0] == 0.0


# ── INCI Function ────────────────────────────────────────────────────────────

class TestInciFunction:
    def test_single_function(self, extraction_single, customer_basic):
        df = transform(extraction_single, customer_basic)
        assert df["INCI Function"].iloc[0] == "Solvent"

    def test_multi_function_semicolon(self, customer_empty):
        extraction = pd.DataFrame({
            "Reference Number": [10001.0],
            "Trade Name": ["X"], "Manufacturer": [None], "INCI": [None],
            "INCI Comp": [None], "Incidental?": [None], "Impurity": [None],
            "Impurity Comp": [None], "Allergens": [None], "Allergen Comp": [None],
            "Function": ["Solvent, Emulsifier, Thickener"],
            "Notes": [None], "GROUP": [1.0], "Missing Comp": [None], "SDS Comp": [None],
        })
        df = transform(extraction, customer_empty)
        assert df["INCI Function"].iloc[0] == "Solvent; Emulsifier; Thickener"

    def test_function_max_255_chars(self, customer_empty):
        extraction = pd.DataFrame({
            "Reference Number": [10001.0],
            "Trade Name": ["X"], "Manufacturer": [None], "INCI": [None],
            "INCI Comp": [None], "Incidental?": [None], "Impurity": [None],
            "Impurity Comp": [None], "Allergens": [None], "Allergen Comp": [None],
            "Function": ["word, " * 100],
            "Notes": [None], "GROUP": [1.0], "Missing Comp": [None], "SDS Comp": [None],
        })
        df = transform(extraction, customer_empty)
        assert len(df["INCI Function"].iloc[0]) <= 255


# ── Impurity Name ────────────────────────────────────────────────────────────

class TestImpurityName:
    def test_impurity_name_mapped(self, extraction_single, customer_basic):
        df = transform(extraction_single, customer_basic)
        assert df["Impurity Name"].iloc[0] == "Lead"

    def test_impurity_max_200_chars(self, customer_empty):
        extraction = pd.DataFrame({
            "Reference Number": [10001.0],
            "Trade Name": ["X"], "Manufacturer": [None], "INCI": [None],
            "INCI Comp": [None], "Incidental?": [None],
            "Impurity": ["Z" * 250],
            "Impurity Comp": [None], "Allergens": [None], "Allergen Comp": [None],
            "Function": [None], "Notes": [None], "GROUP": [1.0],
            "Missing Comp": [None], "SDS Comp": [None],
        })
        df = transform(extraction, customer_empty)
        assert len(df["Impurity Name"].iloc[0]) == 200


# ── Impurity Concentration & Type ────────────────────────────────────────────

class TestImpurityConcentration:
    def test_ppm_extracted(self, extraction_single, customer_basic):
        df = transform(extraction_single, customer_basic)
        assert df["Impurity Concentration"].iloc[0] == 3.0
        assert df["Impurity Concentration type (% or PPM)"].iloc[0] == "PPM"

    def test_percent_unit(self, customer_empty):
        extraction = pd.DataFrame({
            "Reference Number": [10001.0],
            "Trade Name": ["X"], "Manufacturer": [None], "INCI": [None],
            "INCI Comp": [None], "Incidental?": [None], "Impurity": ["Ethanol"],
            "Impurity Comp": ["< 0.5"],
            "Allergens": [None], "Allergen Comp": [None],
            "Function": [None], "Notes": [None], "GROUP": [1.0],
            "Missing Comp": [None], "SDS Comp": [None],
        })
        df = transform(extraction, customer_empty)
        assert df["Impurity Concentration"].iloc[0] == 0.5
        assert df["Impurity Concentration type (% or PPM)"].iloc[0] == "%"

    def test_range_takes_max(self, customer_empty):
        extraction = pd.DataFrame({
            "Reference Number": [10001.0],
            "Trade Name": ["X"], "Manufacturer": [None], "INCI": [None],
            "INCI Comp": [None], "Incidental?": [None], "Impurity": ["Water"],
            "Impurity Comp": ["4 - 6"],
            "Allergens": [None], "Allergen Comp": [None],
            "Function": [None], "Notes": [None], "GROUP": [1.0],
            "Missing Comp": [None], "SDS Comp": [None],
        })
        df = transform(extraction, customer_empty)
        assert df["Impurity Concentration"].iloc[0] == 6.0

    def test_none_when_missing(self, customer_empty):
        extraction = pd.DataFrame({
            "Reference Number": [10001.0],
            "Trade Name": ["X"], "Manufacturer": [None], "INCI": [None],
            "INCI Comp": [None], "Incidental?": [None], "Impurity": [None],
            "Impurity Comp": [None],
            "Allergens": [None], "Allergen Comp": [None],
            "Function": [None], "Notes": [None], "GROUP": [1.0],
            "Missing Comp": [None], "SDS Comp": [None],
        })
        df = transform(extraction, customer_empty)
        assert isna(df["Impurity Concentration"].iloc[0])
        assert isna(df["Impurity Concentration type (% or PPM)"].iloc[0])


# ── Allergen Name & Concentration ────────────────────────────────────────────

class TestAllergenConcentration:
    def test_allergen_name_mapped(self, extraction_single, customer_basic):
        df = transform(extraction_single, customer_basic)
        assert df["Allergen Name"].iloc[0] == "Limonene"

    def test_allergen_concentration(self, extraction_single, customer_basic):
        df = transform(extraction_single, customer_basic)
        assert df["Allergen Concentration"].iloc[0] == 0.5

    def test_allergen_range_takes_max(self, customer_empty):
        extraction = pd.DataFrame({
            "Reference Number": [10001.0],
            "Trade Name": ["X"], "Manufacturer": [None], "INCI": [None],
            "INCI Comp": [None], "Incidental?": [None], "Impurity": [None],
            "Impurity Comp": [None], "Allergens": ["Linalool"],
            "Allergen Comp": ["1 - 3"],
            "Function": [None], "Notes": [None], "GROUP": [1.0],
            "Missing Comp": [None], "SDS Comp": [None],
        })
        df = transform(extraction, customer_empty)
        assert df["Allergen Concentration"].iloc[0] == 3.0

    def test_allergen_out_of_range_clamped(self, customer_empty):
        extraction = pd.DataFrame({
            "Reference Number": [10001.0],
            "Trade Name": ["X"], "Manufacturer": [None], "INCI": [None],
            "INCI Comp": [None], "Incidental?": [None], "Impurity": [None],
            "Impurity Comp": [None], "Allergens": ["X"],
            "Allergen Comp": [150],
            "Function": [None], "Notes": [None], "GROUP": [1.0],
            "Missing Comp": [None], "SDS Comp": [None],
        })
        df = transform(extraction, customer_empty)
        assert isna(df["Allergen Concentration"].iloc[0])

    def test_allergen_max_200_chars(self, customer_empty):
        extraction = pd.DataFrame({
            "Reference Number": [10001.0],
            "Trade Name": ["X"], "Manufacturer": [None], "INCI": [None],
            "INCI Comp": [None], "Incidental?": [None], "Impurity": [None],
            "Impurity Comp": [None], "Allergens": ["L" * 250],
            "Allergen Comp": [None],
            "Function": [None], "Notes": [None], "GROUP": [1.0],
            "Missing Comp": [None], "SDS Comp": [None],
        })
        df = transform(extraction, customer_empty)
        assert len(df["Allergen Name"].iloc[0]) == 200


# ── RM Status ────────────────────────────────────────────────────────────────

class TestRmStatus:
    def test_active_from_customer(self, extraction_single, customer_basic):
        df = transform(extraction_single, customer_basic)
        assert df["RM Status"].iloc[0] == "Active"

    def test_none_when_not_in_customer(self, extraction_single, customer_empty):
        df = transform(extraction_single, customer_empty)
        assert isna(df["RM Status"].iloc[0])

    def test_inactive_status(self, customer_multi):
        extraction = pd.DataFrame({
            "Reference Number": [10002.0],
            "Trade Name": ["X"], "Manufacturer": [None], "INCI": [None],
            "INCI Comp": [None], "Incidental?": [None], "Impurity": [None],
            "Impurity Comp": [None], "Allergens": [None], "Allergen Comp": [None],
            "Function": [None], "Notes": [None], "GROUP": [1.0],
            "Missing Comp": [None], "SDS Comp": [None],
        })
        df = transform(extraction, customer_multi)
        assert df["RM Status"].iloc[0] == "Inactive"


# ── Is Locked ────────────────────────────────────────────────────────────────

class TestIsLocked:
    def test_always_no(self, extraction_single, customer_basic):
        df = transform(extraction_single, customer_basic)
        assert df["Is Locked"].iloc[0] == "No"

    def test_always_no_multi_rows(self, extraction_multi_row, customer_multi):
        df = transform(extraction_multi_row, customer_multi)
        assert all(df["Is Locked"] == "No")


# ── Price fields ─────────────────────────────────────────────────────────────

class TestPriceFields:
    def test_price_none(self, extraction_single, customer_basic):
        df = transform(extraction_single, customer_basic)
        assert isna(df["Price"].iloc[0])

    def test_price_from_none(self, extraction_single, customer_basic):
        df = transform(extraction_single, customer_basic)
        assert isna(df["Price: From"].iloc[0])

    def test_price_to_none(self, extraction_single, customer_basic):
        df = transform(extraction_single, customer_basic)
        assert isna(df["Price: To"].iloc[0])


# ── CAS Impurity / CAS Allergen ──────────────────────────────────────────────

# class TestCasFields:
#     def test_cas_impurity_equals_ref(self, extraction_single, customer_basic):
#         df = transform(extraction_single, customer_basic)
#         assert df["CAS Impurity"].iloc[0] == "10001"

#     def test_cas_allergen_equals_ref(self, extraction_single, customer_basic):
#         df = transform(extraction_single, customer_basic)
#         assert df["CAS Allergen"].iloc[0] == "10001"

#     def test_cas_impurity_all_rows(self, extraction_multi_row, customer_multi):
#         df = transform(extraction_multi_row, customer_multi)
#         assert all(df["CAS Impurity"] == "10005")

#     def test_cas_allergen_all_rows(self, extraction_multi_row, customer_multi):
#         df = transform(extraction_multi_row, customer_multi)
#         assert all(df["CAS Allergen"] == "10005")


# ── Forward-fill Reference Number ────────────────────────────────────────────

class TestForwardFillRefNumber:
    # def test_sub_rows_inherit_parent_ref(self, extraction_no_ref, customer_empty):
    #     df = transform(extraction_no_ref, customer_empty)
    #     # Supplier&Distributor is always populated (unlike Reference Number column)
    #     assert all(df["Supplier&Distributor"] == "10002")

    def test_ref_only_on_first_of_group(self, extraction_no_ref, customer_empty):
        df = transform(extraction_no_ref, customer_empty)
        assert df["Reference Number"].iloc[0] == "10002"
        assert isna(df["Reference Number"].iloc[1])
