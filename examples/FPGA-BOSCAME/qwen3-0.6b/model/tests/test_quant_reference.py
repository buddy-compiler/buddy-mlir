import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from quant_reference import quantize_rows,linear_w8a8,error_metrics


class QuantizationContract(unittest.TestCase):
    def test_zero_signed_half_and_saturation(self):
        values=np.array([[0,0,0,0,0,0],[127,-127,0.5,-0.5,1.5,-1.5]],dtype=np.float32)
        q,s=quantize_rows(values)
        np.testing.assert_array_equal(q,[[0]*6,[127,-127,1,-1,2,-2]])
        np.testing.assert_array_equal(s,[1,1])

    def test_independent_integer_oracle_and_multiply_order(self):
        a=np.array([[1,-2,3],[4,-5,6]],dtype=np.float32)
        w=np.array([[0.13,0.23,-0.54],[-2,0,3]],dtype=np.float32)
        aq,sa=quantize_rows(a);wq,sw=quantize_rows(w)
        result=linear_w8a8(a,wq,sw)
        for row in range(2):
            for col in range(2):
                dot=sum(int(aq[row,k])*int(wq[col,k]) for k in range(3))
                expected=np.float32(np.float32(np.float32(dot)*sa[row])*sw[col])
                self.assertEqual(result[row,col].tobytes(),expected.tobytes())
        metrics=error_metrics(result,a@w.T)
        self.assertGreater(metrics["max_abs_error"],0)

    def test_reject_invalid_contract(self):
        for data in ([[np.nan]],[[np.inf]],[[]],[[np.nextafter(np.float32(0),np.float32(1))]]):
            with self.assertRaises(ValueError):quantize_rows(data)
        with self.assertRaises(ValueError):linear_w8a8([[1]],np.array([[-128]],dtype=np.int8),[1])
        with self.assertRaises(ValueError):error_metrics([np.nan],[0])


if __name__=="__main__":unittest.main()
