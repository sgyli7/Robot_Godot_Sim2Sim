import unittest
import cv2
import numpy as np
from unitree_g1_static_label_fit import apple_label,register


class PrintedLabelAdmissionTests(unittest.TestCase):
    def image(self,marker_id=31,x=100):
        image=np.ones((480,640),np.uint8)*210
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        image[200:260,x:x+60]=cv2.aruco.generateImageMarker(dictionary,marker_id,60)
        image[196:200,x-4:x+64]=255;image[260:264,x-4:x+64]=255
        image[196:264,x-4:x]=255;image[196:264,x+60:x+64]=255
        return image,np.array([[x,200],[x+59,200],[x+59,259],[x,259]],np.float32)

    def test_public_pattern_refines_pixels_and_proves_all_cells(self):
        image,corners=self.image();result=register(image,corners)
        self.assertIsNotNone(result);points,receipt=result
        self.assertEqual(receipt['printed_cell_bit_errors'],0)
        self.assertGreaterEqual(receipt['template_contrast'],.5)
        self.assertLessEqual(np.max(abs(points-corners)),3)

    def test_uniform_quad_cannot_win_through_low_photometric_residual(self):
        _,corners=self.image()
        self.assertIsNone(register(np.ones((480,640),np.uint8)*127,corners))

    def test_wrong_public_dictionary_id_or_corrupt_payload_is_rejected(self):
        image,corners=self.image(32)
        self.assertIsNone(register(image,corners))
        image,corners=self.image();image[210:220,110:120]=255-image[210:220,110:120]
        self.assertIsNone(register(image,corners))

    def test_two_valid_labels_do_not_select_one_as_a_target(self):
        a,ca=self.image(x=100);b,cb=self.image(x=400)
        a[:,396:464]=b[:,396:464];rgb=cv2.cvtColor(a,cv2.COLOR_GRAY2BGR)
        with self.assertRaisesRegex(ValueError,'Multiple actual printed'):
            apple_label(rgb,[ca[None],cb[None]],np.array([[31],[31]]),[])

    def test_too_small_clipped_and_unbounded_candidates_are_rejected(self):
        image,corners=self.image()
        self.assertIsNone(register(image,np.array([[1,1],[7,1],[7,7],[1,7]],float)))
        self.assertIsNone(register(image,corners-[100,0]))
        with self.assertRaisesRegex(ValueError,'candidate budget'):
            apple_label(cv2.cvtColor(image,cv2.COLOR_GRAY2BGR),[],None,[corners]*17)


if __name__=='__main__':unittest.main()
