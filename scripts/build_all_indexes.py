from app.index_builder import build_faiss_index_for_json

build_faiss_index_for_json(
    "products.json",
    "products_index",
    [
        "name",                  # اسم المنتج
        "description",           # وصف المنتج
        "category",              # الفئة (ملابس، أحذية..)
        "brand.displayName",     # اسم البراند
        "brand.description",     # وصف البراند
        "brand.verificationStatus", # حالة توثيق البراند
        "price",                 # السعر
        "discountPercentage",    # نسبة الخصم
        "isCustomizable",        # إمكانية التخصيص
        "reels.videoUrl"         # ربط الريلز (قد تحتوي كلمات مفتاحية)
    ]
)
build_faiss_index_for_json(
    "brands.json",
    "brands_index",
    [
        "displayName",                # اسم العلامة التجارية
        "description",                # وصف العلامة التجارية
        "verificationStatus",         # حالة التوثيق (Verified / Unverified)
        "returnPolicyAsHtml",         # سياسة الاسترجاع مكتوبة داخل HTML
        "products.name",              # أسماء المنتجات التابعة للعلامة
        "products.description",       # وصف المنتجات
        "products.category",          # تصنيف المنتجات (ملابس، أحذية، ... إلخ)
        "products.price",             # السعر (يساعد البحث العددي والنصي)
        "products.discountPercentage",# نسبة الخصم
        "products.isCustomizable",    # هل المنتج قابل للتخصيص
        "products.reels.videoUrl"     # فيديوهات المنتجات (Reels)
    ]
)

build_faiss_index_for_json(
    "reels.json",
    "reels_index",
    [
        "videoUrl",             # رابط الفيديو الأساسي
        "numOfLikes",           # عدد الإعجابات
        "numOfWatches",         # عدد المشاهدات
        "brand.displayName",    # اسم العلامة التجارية المرتبطة
        "brand.description",    # وصف العلامة التجارية
        "brand.verificationStatus", # حالة التوثيق للعلامة
        "product.name",         # اسم المنتج المرتبط بالفيديو
        "product.description",  # وصف المنتج
        "product.category",     # تصنيف المنتج
        "product.price",        # سعر المنتج
        "product.discountPercentage" # نسبة الخصم على المنتج
    ]
)
