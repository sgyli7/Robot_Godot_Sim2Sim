"""Bounded RGB registration of the disclosed 20mm apple label.

Only image pixels and the public DICT_4X4_50 pattern enter this module. Small
tags need full-pattern registration: a one-pixel contour error can corrupt PnP
depth even when its four-corner reprojection error is low. This is classical
perception, not a learned model or task controller.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr

METHOD = 'public_apple_label_full_pattern_registration_v1'
SQUARE = np.array([[-.5,.5],[.5,.5],[.5,-.5],[-.5,-.5]], np.float32)
MAX_REJECTED_QUADS = 16


def register(gray, points):
    """Refine one image quadrilateral; reject uniform, wrong or ambiguous bits."""
    pts = np.asarray(points, dtype=np.float64)
    if (gray.shape != (480,640) or pts.shape != (4,2)
            or not np.isfinite(pts).all()
            or np.linalg.norm(pts-np.roll(pts,-1,axis=0),axis=1).min() < 8):
        return None
    low = np.floor(pts.min(axis=0)-3).astype(int)
    high = np.ceil(pts.max(axis=0)+3).astype(int)
    if (low < 0).any() or (high >= [640,480]).any():
        return None
    if np.prod(high-low+1) > 65536:
        return None
    yy,xx = np.mgrid[low[1]:high[1]+1,low[0]:high[0]+1]
    # Raster samples have half-pixel centers. Returned corners describe the
    # continuous projection plane, retaining the original cx320/cy240 camera.
    coordinates = np.stack([xx.ravel()+.5,yy.ravel()+.5,np.ones(xx.size)])
    observed = gray[yy,xx].ravel()/255.
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    bits = (cv2.aruco.generateImageMarker(dictionary,31,60)[5::10,5::10]>127)
    black = (~bits).astype(np.float64)
    cell_edges = -.5 + np.arange(7)/6
    sigma = .45/np.linalg.norm(pts-np.roll(pts,-1,axis=0),axis=1).mean()

    def prediction(p):
        H = cv2.getPerspectiveTransform(SQUARE,p.reshape(4,2).astype(np.float32))
        uv = np.linalg.inv(H)@coordinates
        u,v = uv[0]/uv[2],-uv[1]/uv[2]
        # The public cells share six horizontal/vertical intervals. Evaluate
        # each interval once and sum the same separable Gaussian coverage;
        # no pattern, antialias kernel or admission threshold changes.
        xcoverage = (ndtr((u[:,None]-cell_edges[:-1])/sigma)
                     - ndtr((u[:,None]-cell_edges[1:])/sigma))
        ycoverage = (ndtr((v[:,None]-cell_edges[:-1])/sigma)
                     - ndtr((v[:,None]-cell_edges[1:])/sigma))
        tone = 1. - ((xcoverage @ black.T) * ycoverage).sum(axis=1)
        valid = (abs(u)<.61)&(abs(v)<.61)
        if valid.sum() < 16:
            raise ValueError('Too few actual label pixels')
        coefficient = np.linalg.lstsq(
            np.stack([tone[valid],np.ones(valid.sum())],axis=1),
            observed[valid],rcond=None)[0]
        residual = (coefficient[0]*tone+coefficient[1]-observed)*valid
        return H,valid,coefficient,residual

    try:
        result = least_squares(lambda p:prediction(p)[3],pts.ravel(),
            bounds=(pts.ravel()-3,pts.ravel()+3),diff_step=.0005,
            max_nfev=100,ftol=1e-7,xtol=1e-7,gtol=1e-7)
        H,valid,coefficient,residual = prediction(result.x)
    except (ValueError,np.linalg.LinAlgError,cv2.error):
        return None
    contrast = float(coefficient[0])
    rms = float(np.sqrt(np.mean(residual[valid]**2)))
    if not result.success or contrast < .5 or rms > .15:
        return None
    samples = []
    for row in range(6):
        for col in range(6):
            uv = np.array([[-.5+(col+.5+dx)/6,.5-(row+.5+dy)/6,1.]
                for dy in [-.15,0,.15] for dx in [-.15,0,.15]]).T
            xy = H@uv
            xy = xy[:2]/xy[2]
            values = cv2.remap(gray.astype(np.float32)/255.,
                (xy[0]-.5).astype(np.float32).reshape(1,-1),
                (xy[1]-.5).astype(np.float32).reshape(1,-1),cv2.INTER_LINEAR)
            samples.append(float(values.mean()))
    normalized = (np.array(samples).reshape(6,6)-coefficient[1])/contrast
    bit_errors = int(np.count_nonzero((normalized>.5)!=bits))
    confidence = float(np.min(abs(normalized-.5)))
    refined = result.x.reshape(4,2)
    if (bit_errors or confidence < .15 or not np.isfinite(refined).all()
            or np.linalg.norm(refined-np.roll(refined,-1,axis=0),axis=1).min()<8):
        return None
    return refined, {'method':METHOD,'template_contrast':contrast,
        'printed_cell_bit_errors':bit_errors,'minimum_bit_confidence':confidence,
        'residual_rms_normalized':rms,'valid_pixels':int(valid.sum()),
        'optimizer_evaluations':result.nfev,'maximum_corner_adjustment_px':3.}


def apple_label(image, corners, ids, rejected):
    """Consider bounded actual contours; an undecoded contour is never an ID."""
    if len(rejected)>MAX_REJECTED_QUADS:
        raise ValueError('Static label candidate budget exceeded')
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    quads = [(c.reshape(4,2),False) for c,i in
             zip(corners,[] if ids is None else ids.flatten()) if int(i)==31]
    quads += [(np.roll(c.reshape(4,2),k,axis=0),True)
              for c in rejected for k in range(4)]
    admitted = []
    for quad,undecoded in quads:
        fit = register(gray,quad)
        if fit is not None:
            points,receipt = fit
            receipt['initial_dictionary_decode_rejected'] = undecoded
            admitted.append((points,receipt))
    if not admitted:
        return None
    best = min(admitted,key=lambda r:r[1]['residual_rms_normalized'])
    if any(np.linalg.norm(p.mean(axis=0)-best[0].mean(axis=0))>2
           for p,_ in admitted):
        raise ValueError('Multiple actual printed apple labels')
    return best
