import pytest

import BlendPATH
from BlendPATH import Global as gl
from BlendPATH.network import pipeline_components as bp_plc


class MakeCompressor:
    x = bp_plc.Composition(pure_x={"CH4": 1})
    p_in = 3000000
    p_out = 6000000
    node1 = bp_plc.Node(name="comp_fm_node", X=x, pressure=p_in)
    node2 = bp_plc.Node(name="comp_to_node", X=x)
    compr = bp_plc.Compressor(
        from_node=node1, to_node=node2, pressure_out_mpa_g=p_out / gl.MPA2PA
    )


class TestCompressor(MakeCompressor):

    def test_compressor_node_pressure_set(self):
        assert self.compr.to_node.pressure == self.p_out

    def test_compression_ratio(self):
        assert self.compr.compression_ratio == self.p_out / self.p_in
