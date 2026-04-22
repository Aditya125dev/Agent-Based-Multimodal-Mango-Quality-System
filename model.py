import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import structural_similarity as ssim


def phash(img):
    img = cv2.resize(img, (32, 32))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    dct_low = dct[:8, :8]

    med = np.median(dct_low)
    return (dct_low > med).flatten()


def hamming_distance(h1, h2):
    return np.sum(h1 != h2)
# ==========================
# LOAD MODELS
# ==========================
rip_model = joblib.load("ripeness_model.pkl")
grd_model = joblib.load("grade_model.pkl")
scaler = joblib.load("scaler.pkl")
rip_enc = joblib.load("ripeness_encoder.pkl")
grd_enc = joblib.load("grade_encoder.pkl")


# ==========================
# SEGMENTATION
# ==========================
def segment_mango(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 30, 30])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mango = cv2.bitwise_and(img, img, mask=mask)
    return mango


# ==========================
# VALIDATION
# ==========================
def check_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < 35, score


def check_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2].mean()
    return not (40 <= brightness <= 220), brightness


# 🔥 FIXED DUPLICATE CHECK (ROTATION-INVARIANT)
def check_exact_duplicate(img1, img2):

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return False, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    good_matches = [m for m in matches if m.distance < 50]

    match_ratio = len(good_matches) / max(len(kp1), len(kp2))

    if match_ratio > 0.3:
        return True, match_ratio

    return False, match_ratio


# ==========================
# SAME MANGO CHECK
import cv2
import numpy as np

def same_mango_check(img1_path, img2_path):

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    img1 = cv2.resize(img1, (300, 300))
    img2 = cv2.resize(img2, (300, 300))

    # ==========================
    # 1. ORB FEATURE MATCH
    # ==========================
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=1000)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    orb_score = 0

    if des1 is not None and des2 is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        good = [m for m in matches if m.distance < 60]
        orb_score = len(good)

    # ==========================
    # 2. COLOR SIMILARITY (KEY)
    # ==========================
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    color_diff = np.mean(np.abs(hsv1 - hsv2))

    # lower = more similar
    color_score = max(0, 100 - color_diff)

    # ==========================
    # 3. HISTOGRAM SIMILARITY
    # ==========================
    hist1 = cv2.calcHist([hsv1], [0,1], None, [50,60], [0,180,0,256])
    hist2 = cv2.calcHist([hsv2], [0,1], None, [50,60], [0,180,0,256])

    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    # ==========================
    # FINAL DECISION
    # ==========================
    # print for debugging
    print(f"ORB: {orb_score}, Color: {color_score:.2f}, Hist: {hist_score:.2f}")

    if (
        orb_score > 10        # relaxed
        or color_score > 70   # strong color match
        or hist_score > 0.7   # histogram match
    ):
        return True, orb_score

    return False, orb_score
# ==========================
# FEATURE EXTRACTION
# ==========================
def extract_features(img):

    img = cv2.resize(img, (224, 224))

    rgb_mean = img.mean(axis=(0, 1))
    rgb_std = img.std(axis=(0, 1))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_mean = hsv.mean(axis=(0, 1))
    hsv_std = hsv.std(axis=(0, 1))

    hue = hsv[:, :, 0]
    hue_hist, _ = np.histogram(hue, bins=16, range=(0, 180))
    hue_hist = hue_hist / (hue_hist.sum() + 1e-10)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_r = (gray / 4).astype(np.uint8)

    glcm = graycomatrix(
        gray_r,
        [1],
        [0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=64,
        symmetric=True,
        normed=True
    )

    texture = [
        graycoprops(glcm, p).mean()
        for p in ["contrast", "homogeneity", "energy", "correlation", "dissimilarity"]
    ]

    return np.hstack([
        rgb_mean, rgb_std,
        hsv_mean, hsv_std,
        hue_hist,
        texture
    ])


# ==========================
# SMART PRICING
# ==========================
def smart_price(grade, ripeness, weight, defect_score):

    if ripeness == "K":
        base = {"P": 2000, "1": 1800, "2": 1500}.get(grade, 1500)
    else:
        base = {"P": 1800, "1": 1600, "2": 1300}.get(grade, 1300)

    adjustment = 0

    if weight > 550:
        adjustment += 100
    elif weight < 350:
        adjustment -= 100

    if defect_score > 0.25:
        adjustment -= 150
    elif defect_score > 0.15:
        adjustment -= 80

    final_price = base + adjustment

    return final_price, {"base": base, "adjustment": adjustment}


# ==========================
# MAIN FUNCTION
# ==========================
def predict_mango_agent(front_path, back_path, weight):

    img1 = cv2.imread(front_path)
    img2 = cv2.imread(back_path)

    if img1 is None or img2 is None:
        return {
            "error": True,
            "message": "❌ Unable to read images. Please upload valid image files."
        }

    img1 = cv2.resize(img1, (224, 224))
    img2 = cv2.resize(img2, (224, 224))

    # 🔍 BLUR
    blur1, s1 = check_blur(img1)
    blur2, s2 = check_blur(img2)
    if blur1 or blur2:
        return {
            "error": True,
            "message": "❌ Blurry images detected. Please upload clear and focused images."
        }

    # 🔆 BRIGHTNESS
    dark1, b1 = check_brightness(img1)
    dark2, b2 = check_brightness(img2)
    if dark1 or dark2:
        return {
            "error": True,
            "message": "❌ Poor lighting detected. Please upload images with proper brightness."
        }

    # 🔁 DUPLICATE
    dup, score = check_exact_duplicate(img1, img2)
    if dup:
        return {
            "error": True,
            "message": "❌ Duplicate images detected. Please upload front and back views of the mango."
        }

    # 🥭 SAME MANGO
    same, score = same_mango_check(front_path, back_path)

    if not same:
        return {
        "error": True,
        "message": f"❌ Different mangoes detected. Please upload images of the same mango."
    }

    # ==========================
    # FEATURES
    # ==========================
    f1 = extract_features(img1)
    f2 = extract_features(img2)
    feat = (f1 + f2) / 2

    X = np.hstack([feat, [weight]]).reshape(1, -1)
    X = scaler.transform(X)

    rip = rip_model.predict(X)[0]
    grd = grd_model.predict(X)[0]

    ripeness = rip_enc.inverse_transform([rip])[0]
    grade = grd_enc.inverse_transform([grd])[0]

    conf = (max(rip_model.predict_proba(X)[0]) +
            max(grd_model.predict_proba(X)[0])) / 2

    # ==========================
    # DEFECT SCORE
    # ==========================
    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

    dark_spots = hsv[:, :, 2] < 50
    dark_density = dark_spots.sum() / dark_spots.size

    sat = hsv[:, :, 1]
    abnormal = ((sat < 20) | (sat > 250))
    abnormal_ratio = abnormal.sum() / abnormal.size

    defect_score = (dark_density + abnormal_ratio) / 2

    # ==========================
    # PRICE
    # ==========================
    price, breakdown = smart_price(grade, ripeness, weight, defect_score)

    return {
        "error": False,
        "ripeness": ripeness,
        "grade": grade,
        "confidence": conf,
        "price": price,
        "price_breakdown": breakdown
    } 