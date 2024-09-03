import pytest

import BlendPATH


class TestExampleFile:
    wangetal = BlendPATH.BlendPATH_scenario(
        casestudy_name="examples/wangetal2018", verbose=False
    )


class TestBlendRatio(TestExampleFile):
    def test_string(self):
        self.wangetal.blendH2(blend="0.5")

    def test_over_100(self):
        with pytest.raises(ValueError):
            self.wangetal.blendH2(blend=50)


class TestDictPrices:
    def test_electricity_dict(self):
        wangetal_1 = BlendPATH.BlendPATH_scenario(
            casestudy_name="examples/wangetal2018",
            verbose=False,
            new_compressors_electric=True,
            existing_compressors_to_electric=True,
            elec_price=0.07,
        )
        lcot_1 = wangetal_1.run_mod("pl")

        wangetal_2 = BlendPATH.BlendPATH_scenario(
            casestudy_name="examples/wangetal2018",
            verbose=False,
            new_compressors_electric=True,
            existing_compressors_to_electric=True,
            elec_price={
                x: 0.07 * 1.025 ** (i - 1) for i, x in enumerate(range(2019, 2020 + 70))
            },
        )

        lcot_2 = wangetal_2.run_mod("pl")
        assert lcot_1 == pytest.approx(lcot_2)

    def test_ng_dict(self):
        wangetal_1 = BlendPATH.BlendPATH_scenario(
            casestudy_name="examples/wangetal2018",
            verbose=False,
            new_compressors_electric=False,
            existing_compressors_to_electric=False,
            ng_price=7.39,
        )
        lcot_1 = wangetal_1.run_mod("dr")

        wangetal_2 = BlendPATH.BlendPATH_scenario(
            casestudy_name="examples/wangetal2018",
            verbose=False,
            new_compressors_electric=False,
            existing_compressors_to_electric=False,
            ng_price={
                x: 7.39 * 1.025 ** (i - 1) for i, x in enumerate(range(2019, 2020 + 70))
            },
        )

        lcot_2 = wangetal_2.run_mod("dr")
        assert lcot_1 == pytest.approx(lcot_2)


class TestFinancialOverrides:
    def test_d2e(self):
        wangetal = BlendPATH.BlendPATH_scenario(
            casestudy_name="examples/wangetal2018",
            verbose=False,
            financial_overrides={"debt equity ratio of initial financing": 1.5},
            design_option="nfc",
            blend=0.5,
        )
        assert wangetal.run_mod("pl") == pytest.approx(0.349012830807369)

    def test_invalid_financial_override_name(self):
        with pytest.raises(ValueError):
            wangetal = BlendPATH.BlendPATH_scenario(
                casestudy_name="examples/wangetal2018",
                verbose=False,
                financial_overrides={"asdfasdf": 1.5},
                design_option="nfc",
                blend=0.5,
            )
            wangetal.run_mod("pl")


class TestWangNFC(TestExampleFile):
    def test_wang_50_dr_nfc(self):
        self.wangetal.update_design_option(design_option="nfc")
        self.wangetal.blendH2(blend=0.5)
        lcot = self.wangetal.run_mod("dr")
        assert lcot == pytest.approx(0.4314880034390952)

    def test_wang_50_pl_nfc(self):
        lcot = self.wangetal.run_mod("pl")
        assert lcot == pytest.approx(0.392927081805103)

    def test_wang_50_ac_nfc(self):
        lcot = self.wangetal.run_mod("ac")
        assert lcot == pytest.approx(0.698920145455941)


class TestDesignOption(TestExampleFile):
    def test_design_option_bad_string(self):
        with pytest.raises(ValueError):
            self.wangetal.update_design_option(design_option="abcd")

    def test_design_option_not_fraction(self):
        with pytest.raises(ValueError):
            self.wangetal.update_design_option(design_option=50)

    def test_design_option_match_b(self):
        self.wangetal.update_design_option(design_option="b")
        lcot_b = self.wangetal.run_mod("dr")
        self.wangetal.update_design_option(design_option=0.72)
        lcot_72 = self.wangetal.run_mod("dr")
        assert lcot_b == pytest.approx(lcot_72)
