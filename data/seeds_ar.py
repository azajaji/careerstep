"""Hand-authored Arabic CV and job-description seed corpus."""

from __future__ import annotations

from typing import List, Dict, Any


# ============================================================================
# Resumes (30) -- 6 per role
# ============================================================================

_RESUMES_AR: List[Dict[str, str]] = [
    # ---------- data_scientist ----------
    {"id": "ar_ds_1", "category": "data_scientist", "text":
        "نبذة شخصية\n"
        "عالم بيانات بخبرة ثلاث سنوات في بناء نماذج التعلم الآلي على البيانات الجدولية.\n\n"
        "المهارات\n"
        "بايثون، pandas، scikit-learn، الانحدار اللوجستي، أشجار القرار، SQL، Tableau.\n\n"
        "الخبرة\n"
        "طورت نموذج تنبؤ بالاحتفاظ بالعملاء حقق دقة 0.81 على مجموعة الاختبار، ونشرته خلف خدمة REST.\n"
        "بنيت لوحات معلومات تنفيذية في Tableau لقياس المؤشرات الأسبوعية.\n"},
    {"id": "ar_ds_2", "category": "data_scientist", "text":
        "نبذة\n"
        "عالم بيانات حديث التخرج من جامعة الملك سعود، شغوف بالنمذجة الإحصائية وتحليل البيانات.\n\n"
        "المهارات\n"
        "بايثون، numpy، الإحصاء التطبيقي، تنقية البيانات، الانحدار، A/B testing، Git.\n\n"
        "الخبرة\n"
        "أجريت تحليلاً إحصائياً لأكثر من 50 ألف معاملة تجارية واستخرجت أنماط الاحتيال.\n"
        "تدربت في مختبر التعلم الآلي لمدة فصلين.\n"},
    {"id": "ar_ds_3", "category": "data_scientist", "text":
        "ملخص\n"
        "متخصص في التعلم العميق ومعالجة اللغة الطبيعية، مع تركيز على اللغة العربية.\n\n"
        "المهارات\n"
        "PyTorch، transformers، NLP، التضمين الدلالي، تصنيف النصوص، Docker.\n\n"
        "الخبرة\n"
        "حسّنت نموذج تصنيف للمراجعات العربية ورفعت الـ F1 من 0.74 إلى 0.86.\n"
        "نشرت النموذج عبر حاويات Docker على بيئة الإنتاج.\n"},
    {"id": "ar_ds_4", "category": "data_scientist", "text":
        "نبذة\n"
        "محلل بيانات بمسار صاعد نحو علم البيانات، خبرة سنتان في تحليل البيانات التجارية.\n\n"
        "المهارات\n"
        "Excel، SQL، Power BI، الإحصاء الأساسي، تنظيف البيانات، التحليل الاستكشافي.\n\n"
        "الخبرة\n"
        "بنيت تقارير شهرية للإدارة العليا غطّت 12 وحدة أعمال.\n"
        "أتمتت سير عمل تنظيف البيانات لخمسة مصادر OLTP.\n"},
    {"id": "ar_ds_5", "category": "data_scientist", "text":
        "ملخص\n"
        "مهندس تعلم آلي شارك في تطوير منصة MLOps لشركة فنتك في الرياض.\n\n"
        "المهارات\n"
        "Python، MLflow، Apache Airflow، نشر النماذج، مراقبة الإنتاج، AWS.\n\n"
        "الخبرة\n"
        "أتمتت دورة حياة النماذج من التدريب إلى الإنتاج إلى المراقبة.\n"
        "صممت بنية موحدة لإطلاق أكثر من 12 نموذجاً.\n"},
    {"id": "ar_ds_6", "category": "data_scientist", "text":
        "نبذة\n"
        "عالم بيانات مهتم بالتطبيقات الصحية والتنبؤ بسلاسل الزمن.\n\n"
        "المهارات\n"
        "بايثون، scikit-learn، التحليل الزمني، XGBoost، التصور البياني، إحصاء.\n\n"
        "الخبرة\n"
        "طورت نموذج تنبؤ بحالات الطوارئ في المستشفى وحققت تحسناً قدره 15% فوق المعيار.\n"
        "قدمت نتائج التحليل أسبوعياً لفريق متعدد التخصصات.\n"},

    # ---------- software_engineer ----------
    {"id": "ar_se_1", "category": "software_engineer", "text":
        "ملخص\n"
        "مهندس برمجيات بخبرة أربع سنوات في تطوير الخدمات الخلفية والبنية التحتية السحابية.\n\n"
        "المهارات\n"
        "Java، Spring Boot، PostgreSQL، Docker، Kubernetes، REST، Git، CI/CD.\n\n"
        "الخبرة\n"
        "بنيت خدمة مصغرة للمدفوعات تخدم مليون طلب شهرياً.\n"
        "حسنت زمن الاستجابة p99 بنسبة 35% عبر تحسينات الاستعلامات والفهرسة.\n"},
    {"id": "ar_se_2", "category": "software_engineer", "text":
        "نبذة\n"
        "مطور خلفية شغوف بالأنماط المعمارية النظيفة وجودة الكود.\n\n"
        "المهارات\n"
        "Python، Django، REST API، اختبارات الوحدة، TDD، PostgreSQL، Linux.\n\n"
        "الخبرة\n"
        "رفعت تغطية الاختبارات من 32% إلى 84% خلال ربعين.\n"
        "صممت طبقة إذونات قائمة على الأدوار للنظام الأكاديمي.\n"},
    {"id": "ar_se_3", "category": "software_engineer", "text":
        "ملخص\n"
        "مهندس برمجيات حديث التخرج، شارك في مشاريع مفتوحة المصدر منذ السنة الثانية.\n\n"
        "المهارات\n"
        "C++، خوارزميات، هياكل البيانات، Git، Linux، Bash، أساسيات الشبكات.\n\n"
        "الخبرة\n"
        "ساهمت في مكتبة استدلال نموذج التعلم الآلي مفتوحة المصدر بأكثر من 30 ميرج.\n"
        "نظمت ورشة جامعية حول هياكل البيانات لطلاب السنة الأولى.\n"},
    {"id": "ar_se_4", "category": "software_engineer", "text":
        "نبذة\n"
        "مطور بنية تحتية متخصص في الحوسبة السحابية والأتمتة.\n\n"
        "المهارات\n"
        "AWS، Terraform، Ansible، Docker، Kubernetes، Linux، Bash، مراقبة الإنتاج.\n\n"
        "الخبرة\n"
        "هاجرت ثلاث خدمات إنتاج من سيرفرات تقليدية إلى منصة Kubernetes.\n"
        "خفضت تكاليف البنية التحتية بنسبة 28% عبر التحجيم التلقائي.\n"},
    {"id": "ar_se_5", "category": "software_engineer", "text":
        "ملخص\n"
        "مهندس برمجيات بتركيز على الأنظمة الموزعة وقوائم الرسائل.\n\n"
        "المهارات\n"
        "Go، Kafka، gRPC، Redis، PostgreSQL، Prometheus، Grafana.\n\n"
        "الخبرة\n"
        "صممت بنية معالجة الأحداث في الوقت الفعلي لمنصة تجارة إلكترونية.\n"
        "نفذت اختبارات الحمل وقدمت تقارير الأداء للقيادة الفنية.\n"},
    {"id": "ar_se_6", "category": "software_engineer", "text":
        "نبذة\n"
        "مهندس برمجيات للجوال، خبرة بنشر تطبيقات على متاجر التطبيقات لأكثر من مئة ألف مستخدم.\n\n"
        "المهارات\n"
        "Flutter، Dart، REST، Firebase، اختبار الواجهات، Git.\n\n"
        "الخبرة\n"
        "أطلقت تطبيقاً للخدمات الحكومية وصل تقييم 4.6 نجوم.\n"
        "نفذت طبقة المزامنة بين قاعدة البيانات المحلية والسحابية.\n"},

    # ---------- frontend_developer ----------
    {"id": "ar_fe_1", "category": "frontend_developer", "text":
        "ملخص\n"
        "مطور واجهات بخبرة ثلاث سنوات في بناء واجهات سريعة وقابلة للوصول.\n\n"
        "المهارات\n"
        "React، TypeScript، Tailwind، Next.js، اختبار الواجهات، WCAG، Figma.\n\n"
        "الخبرة\n"
        "أعدت كتابة لوحة تحكم العميل بإطار React بأداء تحسن 45% في وقت التحميل.\n"
        "أعدت تصميم 12 شاشة بناءً على نتائج اختبارات المستخدم.\n"},
    {"id": "ar_fe_2", "category": "frontend_developer", "text":
        "نبذة\n"
        "مطور واجهات حديث التخرج، شغوف بنظم التصميم وتجربة المستخدم.\n\n"
        "المهارات\n"
        "JavaScript، Vue.js، CSS، Sass، HTML5، الوصولية، اختبارات الوحدة.\n\n"
        "الخبرة\n"
        "بنيت تطبيق صفحة واحدة لإدارة المهام لمشروع التخرج.\n"
        "ساهمت في مكتبة مكونات الفريق بأكثر من 20 مكوناً.\n"},
    {"id": "ar_fe_3", "category": "frontend_developer", "text":
        "ملخص\n"
        "مطور واجهات بخبرة في تطبيقات التعاون في الوقت الحقيقي.\n\n"
        "المهارات\n"
        "React، Redux، WebSocket، WebRTC، TypeScript، أداء المتصفح، اختبار End-to-End.\n\n"
        "الخبرة\n"
        "نفذت تجربة تحرير جماعي لمستندات تشبه Google Docs.\n"
        "خفضت حجم حزمة JavaScript بنسبة 38% عبر التقسيم الديناميكي.\n"},
    {"id": "ar_fe_4", "category": "frontend_developer", "text":
        "نبذة\n"
        "مطور واجهات يدمج بين تصميم تجربة المستخدم والبرمجة.\n\n"
        "المهارات\n"
        "HTML، CSS، JavaScript، Figma، التصميم الجرافيكي، الرسوم المتحركة، إمكانية الوصول.\n\n"
        "الخبرة\n"
        "صممت ونفذت تجربة تعليمية تفاعلية لمنصة ناشئة في القطاع التعليمي.\n"
        "أجريت 14 جلسة اختبار قابلية استخدام مع طلاب جامعيين.\n"},
    {"id": "ar_fe_5", "category": "frontend_developer", "text":
        "ملخص\n"
        "مطور Angular بخبرة سنتين في تطبيقات الأعمال داخل المؤسسات.\n\n"
        "المهارات\n"
        "Angular، TypeScript، RxJS، NgRx، اختبارات الواجهات، CSS، Git.\n\n"
        "الخبرة\n"
        "ساهمت في إعادة كتابة بوابة الموظف من Angular 8 إلى 16.\n"
        "نظمت ورش عمل داخلية حول أفضل ممارسات RxJS.\n"},
    {"id": "ar_fe_6", "category": "frontend_developer", "text":
        "نبذة\n"
        "مطور واجهات للجوال والويب، يهتم بالأنظمة العابرة للمنصات.\n\n"
        "المهارات\n"
        "React Native، React، JavaScript، Expo، REST، CI/CD.\n\n"
        "الخبرة\n"
        "أطلقت تطبيق مطعم لقياس رضا العملاء بمتوسط 12 ألف زيارة يومية.\n"
        "أتمتت إطلاق إصدارات Beta للمتجرين عبر EAS.\n"},

    # ---------- cybersecurity_analyst ----------
    {"id": "ar_cs_1", "category": "cybersecurity_analyst", "text":
        "ملخص\n"
        "محلل أمن سيبراني بخبرة ثلاث سنوات في مراكز عمليات الأمن.\n\n"
        "المهارات\n"
        "SIEM، Splunk، Wireshark، تحليل البرمجيات الخبيثة، استجابة الحوادث، إطار MITRE ATT&CK.\n\n"
        "الخبرة\n"
        "حققت في أكثر من 200 حادث أمني وقدمت تقارير ما بعد الحادث.\n"
        "ضبطت قواعد كشف خفضت الإنذارات الكاذبة بنسبة 40%.\n"},
    {"id": "ar_cs_2", "category": "cybersecurity_analyst", "text":
        "نبذة\n"
        "متخصص في اختبار الاختراق على تطبيقات الويب والشبكات.\n\n"
        "المهارات\n"
        "Burp Suite، nmap، Metasploit، OWASP Top 10، تقييم نقاط الضعف، Kali Linux.\n\n"
        "الخبرة\n"
        "أجريت 18 اختبار اختراق على تطبيقات داخلية واكتشفت 4 ثغرات حرجة.\n"
        "كتبت تقارير معتمدة بصيغة CVSS للقيادة الفنية.\n"},
    {"id": "ar_cs_3", "category": "cybersecurity_analyst", "text":
        "ملخص\n"
        "محلل أمن حاصل على CompTIA Security+ ويعمل على ISC2 CCSP.\n\n"
        "المهارات\n"
        "TCP/IP، Wireshark، تحليل السجلات، Linux، Python، أتمتة الأمن.\n\n"
        "الخبرة\n"
        "أتمتت تنقية السجلات في خط أنابيب Splunk وفّر نحو 10 ساعات أسبوعياً.\n"
        "ساهمت في تدقيق ISO 27001 الداخلي للشركة.\n"},
    {"id": "ar_cs_4", "category": "cybersecurity_analyst", "text":
        "نبذة\n"
        "محلل أمن خلوي يركز على البنية التحتية السحابية.\n\n"
        "المهارات\n"
        "AWS، Azure، GuardDuty، CloudTrail، التشفير، إدارة الهوية، Terraform.\n\n"
        "الخبرة\n"
        "صممت سياسات الوصول الأقل امتيازاً لأكثر من 60 خدمة AWS.\n"
        "نفذت تشفير البيانات أثناء السكون عبر جميع مخازن S3.\n"},
    {"id": "ar_cs_5", "category": "cybersecurity_analyst", "text":
        "ملخص\n"
        "محلل أمن حديث التخرج ضمن برنامج التدريب الوطني للأمن السيبراني.\n\n"
        "المهارات\n"
        "أساسيات الشبكات، Kali Linux، CTF، Wireshark، CISSP في طور التحضير.\n\n"
        "الخبرة\n"
        "فزت بالمركز الثالث في مسابقة CTF الجامعية لعام 2024.\n"
        "تدربت في فريق SOC لمدة فصل دراسي كامل.\n"},
    {"id": "ar_cs_6", "category": "cybersecurity_analyst", "text":
        "نبذة\n"
        "متخصص في تحليل التهديدات والاستخبارات السيبرانية.\n\n"
        "المهارات\n"
        "Threat Intelligence، MISP، YARA، تحليل البرمجيات الخبيثة، Python، Splunk.\n\n"
        "الخبرة\n"
        "كتبت 30+ قاعدة YARA لاكتشاف عائلات برمجية خبيثة جديدة.\n"
        "نشرت تقارير تهديدات أسبوعية لشركاء القطاع المالي.\n"},

    # ---------- product_manager ----------
    {"id": "ar_pm_1", "category": "product_manager", "text":
        "ملخص\n"
        "مدير منتج بخبرة أربع سنوات في منتجات SaaS للمؤسسات.\n\n"
        "المهارات\n"
        "اكتشاف المنتج، خرائط الطريق، A/B testing، تحليلات SQL، التواصل مع أصحاب المصلحة.\n\n"
        "الخبرة\n"
        "أطلقت أربع ميزات رئيسية أسهمت في نمو الإيرادات السنوية بنسبة 22%.\n"
        "أدرت فريق مشترك مع التصميم والهندسة عبر سبعة قطاعات أعمال.\n"},
    {"id": "ar_pm_2", "category": "product_manager", "text":
        "نبذة\n"
        "مدير منتج تقني يربط بين التحليل الكمي ورؤية المستخدم.\n\n"
        "المهارات\n"
        "Mixpanel، SQL، Figma، اكتشاف المستخدم، إدارة الإصدار، Jira، OKRs.\n\n"
        "الخبرة\n"
        "حدّدت نقاط الاحتكاك في رحلة الإعداد مما رفع التحويل بنسبة 18%.\n"
        "نظمت اجتماعات Quarterly Business Review للقيادة العليا.\n"},
    {"id": "ar_pm_3", "category": "product_manager", "text":
        "ملخص\n"
        "مدير منتج للجوال يركز على تجارب المستخدم في تطبيقات الفنتك.\n\n"
        "المهارات\n"
        "App Store Analytics، A/B testing، إدارة دورة الحياة، اكتشاف المنتج، التحليل التنافسي.\n\n"
        "الخبرة\n"
        "أعدت إطلاق رحلة التسجيل وحققت تحسناً قدره 31% في معدل التحويل.\n"
        "أدرت 22 إطلاقاً متتالياً عبر قنوات A/B testing.\n"},
    {"id": "ar_pm_4", "category": "product_manager", "text":
        "نبذة\n"
        "مدير منتج حديث، انتقل من البرمجة إلى إدارة المنتج بعد ثلاث سنوات هندسة.\n\n"
        "المهارات\n"
        "JIRA، Confluence، SQL، اكتشاف المستخدم، خرائط التأثير، ROI.\n\n"
        "الخبرة\n"
        "حددت خمسة محاور منتج لمشروع التخرج وقدّمت الأطروحة لجنة الكلية.\n"
        "تدربت كمدير منتج لفصلين في شركة ناشئة تعليمية.\n"},
    {"id": "ar_pm_5", "category": "product_manager", "text":
        "ملخص\n"
        "مدير منتج بيانات يبني منتجات قائمة على التعلم الآلي.\n\n"
        "المهارات\n"
        "ML product strategy، SQL، Mixpanel، إدارة أصحاب المصلحة، تقدير العائد.\n\n"
        "الخبرة\n"
        "أطلقت محرك توصية أسهم في زيادة الإيرادات بنسبة 14%.\n"
        "حددت معايير القبول لأكثر من خمسة نماذج تعلم آلي قبل الإطلاق.\n"},
    {"id": "ar_pm_6", "category": "product_manager", "text":
        "نبذة\n"
        "مدير منتج للنمو يتقن أطر اكتساب المستخدمين.\n\n"
        "المهارات\n"
        "Growth Loops، Funnel Analysis، A/B testing، SEO، تحليلات الويب، أتمتة التسويق.\n\n"
        "الخبرة\n"
        "صممت حلقة نمو دفعت 60% من المستخدمين الجدد عبر الإحالة.\n"
        "ضاعفت العضويات النشطة الشهرية في تسعة أشهر.\n"},
]


# ============================================================================
# CV / JD pairs (30 matched + 5 distractors per role for retrieval)
# ============================================================================

_JDS_AR: Dict[str, List[str]] = {
    "data_scientist": [
        "نبحث عن عالم بيانات لتطوير نماذج التعلم الآلي على بيانات الأعمال. متطلبات: بايثون، pandas، scikit-learn، SQL، خبرة في النمذجة الإحصائية، التواصل مع غير التقنيين.",
        "نطاق العمل يشمل بناء نماذج التنبؤ، تنظيف البيانات، تطوير لوحات المعلومات، واختبار A/B. خبرة 2-4 سنوات في علم البيانات أو التحليل المتقدم.",
        "وظيفة عالم بيانات أول لفريق NLP. مطلوب PyTorch، transformers، خبرة بمعالجة النصوص العربية أو متعددة اللغات، نشر النماذج في الإنتاج.",
        "محلل بيانات/عالم بيانات بمستوى متوسط لشركة فنتك. SQL وExcel متقدم، Power BI، التحليل التجاري، خبرة بقطاع المدفوعات أو الإقراض.",
        "مهندس تعلم آلي لمنصة MLOps. مطلوب MLflow، Airflow، AWS، نشر النماذج، مراقبة الإنتاج، خبرة بأنظمة متكاملة.",
        "عالم بيانات صحي. مطلوب التنبؤ بالسلاسل الزمنية، XGBoost، التواصل مع الفرق الطبية، فهم البيانات الصحية، Python.",
    ],
    "software_engineer": [
        "مهندس برمجيات خلفية بخبرة 3-5 سنوات. Java أو Python، PostgreSQL، Docker، Kubernetes، REST، تجربة CI/CD.",
        "مطور Python خلفية مع تركيز على جودة الكود. Django أو Flask، اختبارات الوحدة، TDD، PostgreSQL.",
        "مهندس برمجيات شامل مفتوح المصدر. C++ أو Rust، خوارزميات، Linux، Git، اهتمام بالمساهمة في مشاريع OSS.",
        "مهندس بنية تحتية. AWS، Terraform، Ansible، Kubernetes، خبرة هجرة الخدمات إلى السحابة.",
        "مهندس برمجيات للأنظمة الموزعة. Go أو Java، Kafka، gRPC، Redis، Prometheus، خبرة بالأحداث في الوقت الحقيقي.",
        "مطور تطبيقات جوال. Flutter وDart، نشر على المتجرين، Firebase، اختبارات الواجهات.",
    ],
    "frontend_developer": [
        "مطور واجهات React بخبرة 2-4 سنوات. TypeScript، Tailwind، Next.js، WCAG، تكامل مع REST.",
        "مطور واجهات حديث التخرج لفريق منتجات SaaS. JavaScript، Vue.js، CSS، اختبارات الواجهات.",
        "مطور React متقدم لتطبيقات تعاون في الوقت الحقيقي. Redux، WebSocket، WebRTC، أداء المتصفح.",
        "مطور UX/Frontend هجين. HTML/CSS/JS، Figma، إجراء جلسات اختبار قابلية الاستخدام.",
        "مطور Angular لتطبيقات أعمال المؤسسات. Angular 14+، RxJS، NgRx، اختبارات الواجهات.",
        "مطور React Native للجوال. تطبيقات عابرة للمنصات، Expo، REST.",
    ],
    "cybersecurity_analyst": [
        "محلل أمن سيبراني لمركز عمليات الأمن. SIEM (Splunk)، تحليل السجلات، استجابة الحوادث، خبرة 2+ سنوات.",
        "محلل اختبار اختراق. Burp Suite، nmap، OWASP Top 10، Kali Linux، تقارير CVSS.",
        "محلل أمن مبتدئ مع شهادة Security+. أساسيات الشبكات، Wireshark، Python، Linux.",
        "محلل أمن سحابي. AWS Security، GuardDuty، CloudTrail، Terraform، إدارة الهوية.",
        "محلل أمن لتدريب نهاية البرنامج. خريج جديد، CTF، أساسيات الأمن، شغف للتعلم.",
        "محلل تهديدات سيبرانية. MISP، YARA، تحليل البرمجيات الخبيثة، تقارير التهديدات للقطاع المالي.",
    ],
    "product_manager": [
        "مدير منتج SaaS للمؤسسات. اكتشاف المنتج، خرائط الطريق، A/B testing، تحليلات SQL، إدارة أصحاب المصلحة، خبرة 3-5 سنوات.",
        "مدير منتج تقني. Mixpanel، SQL، Figma، Jira، اكتشاف المستخدم، إدارة الإصدار، OKRs.",
        "مدير منتج جوال للفنتك. App Store Analytics، A/B testing، اكتشاف المنتج، التحليل التنافسي.",
        "مدير منتج مبتدئ (انتقال من الهندسة). JIRA، Confluence، SQL، اكتشاف المستخدم، تواصل ممتاز.",
        "مدير منتج بيانات/تعلم آلي. ML product strategy، SQL، إدارة دورة حياة النماذج، فهم البيانات.",
        "مدير منتج للنمو. Growth Loops، Funnel Analysis، A/B testing، SEO، أتمتة التسويق.",
    ],
}


def resumes_ar() -> List[Dict[str, str]]:
    return list(_RESUMES_AR)


def resume_jd_pairs_ar() -> List[Dict[str, str]]:
    """One Arabic CV paired with one Arabic JD per resume (matched by role)."""
    out: List[Dict[str, str]] = []
    for resume in _RESUMES_AR:
        role = resume["category"]
        # Round-robin assign a JD from the role's pool by index suffix.
        idx = int(resume["id"].split("_")[-1]) - 1
        jd_pool = _JDS_AR[role]
        jd = jd_pool[idx % len(jd_pool)]
        out.append({"resume": resume["text"], "job_description": jd, "role": role})
    return out


# ============================================================================
# Arabic paraphrase pairs (50 items, ESCO english taxonomy -> Arabic CV phrase)
# ============================================================================

_PARAPHRASE_AR: List[Dict[str, Any]] = [
    # technical positives
    {"taxonomy": "Python",                 "cv_phrase": "كتبت أدوات تحليل بيانات بلغة بايثون",         "label": 1, "skill_type": "technical"},
    {"taxonomy": "SQL",                    "cv_phrase": "صممت استعلامات بقاعدة بيانات علائقية",        "label": 1, "skill_type": "technical"},
    {"taxonomy": "REST API",               "cv_phrase": "بنيت خدمات ويب بنمط نقاط النهاية REST",        "label": 1, "skill_type": "technical"},
    {"taxonomy": "Docker",                 "cv_phrase": "حزمت الخدمة في حاويات Docker",                 "label": 1, "skill_type": "technical"},
    {"taxonomy": "Kubernetes",             "cv_phrase": "أدرت Pods وDeployments على منصة k8s",          "label": 1, "skill_type": "technical"},
    {"taxonomy": "machine learning",       "cv_phrase": "دربت نماذج تعلم آلي على بيانات جدولية",        "label": 1, "skill_type": "technical"},
    {"taxonomy": "deep learning",          "cv_phrase": "ضبّطت نماذج التعلم العميق المسبقة التدريب",     "label": 1, "skill_type": "technical"},
    {"taxonomy": "natural language processing", "cv_phrase": "بنيت مصنّفاً للنصوص العربية",              "label": 1, "skill_type": "technical"},
    {"taxonomy": "AWS",                    "cv_phrase": "نشرت الخدمات على البنية السحابية لأمازون",     "label": 1, "skill_type": "technical"},
    {"taxonomy": "Azure",                  "cv_phrase": "أدرت تطبيقات على منصة مايكروسوفت السحابية",     "label": 1, "skill_type": "technical"},
    {"taxonomy": "Linux",                  "cv_phrase": "أشغّل البنية الإنتاجية على توزيعات لينكس",      "label": 1, "skill_type": "technical"},
    {"taxonomy": "Git",                    "cv_phrase": "أعمل بنظام تحكم في الإصدارات الموزع",           "label": 1, "skill_type": "technical"},
    {"taxonomy": "React",                  "cv_phrase": "بنيت تطبيق صفحة واحدة بإطار React",            "label": 1, "skill_type": "technical"},
    {"taxonomy": "TypeScript",             "cv_phrase": "كتبت كود الإنتاج بنكهة JavaScript المُنمَّطة",   "label": 1, "skill_type": "technical"},
    {"taxonomy": "PostgreSQL",             "cv_phrase": "ضبطت الفهارس على قاعدة بيانات Postgres",        "label": 1, "skill_type": "technical"},
    {"taxonomy": "CI/CD",                  "cv_phrase": "أنشأت خطوط بناء واختبار ونشر آلية",            "label": 1, "skill_type": "technical"},
    {"taxonomy": "Apache Spark",           "cv_phrase": "وسّعت خط ETL إلى أحجام تيرابايت بحوسبة موزعة",  "label": 1, "skill_type": "technical"},
    {"taxonomy": "OWASP Top 10",           "cv_phrase": "حصّنت التطبيق ضد ثغرات الويب الشائعة",          "label": 1, "skill_type": "technical"},
    {"taxonomy": "penetration testing",    "cv_phrase": "نفذت اختبارات اختراق مرخّصة على البيئة الإنتاجية", "label": 1, "skill_type": "technical"},
    {"taxonomy": "Wireshark",              "cv_phrase": "حللت حزم الشبكة لعزل مشكلات الكمون",            "label": 1, "skill_type": "technical"},
    {"taxonomy": "ETL",                    "cv_phrase": "نقلت البيانات من المصادر إلى مستودع البيانات",  "label": 1, "skill_type": "technical"},
    {"taxonomy": "A/B testing",            "cv_phrase": "صممت تجارب عشوائية لمسارات المستخدم",           "label": 1, "skill_type": "technical"},
    {"taxonomy": "Tableau",                "cv_phrase": "بنيت لوحات معلومات بأداة سحب وإفلات",           "label": 1, "skill_type": "technical"},
    {"taxonomy": "Power BI",               "cv_phrase": "صمّمت مقاييس DAX على نموذج بيانات متعدد الجداول", "label": 1, "skill_type": "technical"},
    {"taxonomy": "Figma",                  "cv_phrase": "أعمل على نظام التصميم في أداة تصميم متجهية تعاونية", "label": 1, "skill_type": "technical"},
    {"taxonomy": "MongoDB",                "cv_phrase": "نمذجت كتالوج المنتج في مخزن مستندات NoSQL",     "label": 1, "skill_type": "technical"},
    {"taxonomy": "Apache Airflow",         "cv_phrase": "نظمت مهام دفعية ليلية في مخطط DAG",             "label": 1, "skill_type": "technical"},
    {"taxonomy": "data visualisation",     "cv_phrase": "قدّمت رؤى البيانات عبر رسوم تفاعلية",            "label": 1, "skill_type": "technical"},
    {"taxonomy": "feature engineering",    "cv_phrase": "اشتققت متغيرات ذات نوافذ زمنية متحركة",         "label": 1, "skill_type": "technical"},
    {"taxonomy": "model deployment",       "cv_phrase": "أنزلت النموذج المدرّب خلف واجهة برمجة منخفضة الكمون", "label": 1, "skill_type": "technical"},

    # soft positives
    {"taxonomy": "communication",          "cv_phrase": "قدّمت تحديثات أسبوعية لقيادة غير تقنية",         "label": 1, "skill_type": "soft"},
    {"taxonomy": "teamwork",               "cv_phrase": "شاركت في قيادة فرقة عمل متعددة التخصصات",      "label": 1, "skill_type": "soft"},
    {"taxonomy": "leadership",             "cv_phrase": "وجّهت ثلاثة مهندسين مبتدئين عبر مشروع كبير",     "label": 1, "skill_type": "soft"},
    {"taxonomy": "problem solving",        "cv_phrase": "شخّصت مشكلة كمون استمرت أشهراً خلال يومين",     "label": 1, "skill_type": "soft"},
    {"taxonomy": "time management",        "cv_phrase": "سلّمت بشكل متسق في أو قبل المواعيد المحددة",     "label": 1, "skill_type": "soft"},
    {"taxonomy": "stakeholder management", "cv_phrase": "وحّدت أصحاب المصلحة في الفرق المتعددة على خطة الإصدار", "label": 1, "skill_type": "soft"},
    {"taxonomy": "mentoring",              "cv_phrase": "أدرت اجتماعات أسبوعية فردية مع متدربين",       "label": 1, "skill_type": "soft"},
    {"taxonomy": "Arabic / English bilingual", "cv_phrase": "أكتب وثائق العملاء بالعربية والإنجليزية",  "label": 1, "skill_type": "soft"},

    # cert positives
    {"taxonomy": "AWS Certified Solutions Architect", "cv_phrase": "حاصل على شهادة AWS SAA-C03 سارية",   "label": 1, "skill_type": "cert"},
    {"taxonomy": "Microsoft Certified: Azure Fundamentals (AZ-900)", "cv_phrase": "اجتزت اختبار AZ-900", "label": 1, "skill_type": "cert"},
    {"taxonomy": "PMP",                    "cv_phrase": "حاصل على شهادة محترف إدارة المشاريع PMI",      "label": 1, "skill_type": "cert"},
    {"taxonomy": "CompTIA Security+",      "cv_phrase": "اجتزت اختبار CompTIA Security+ SY0-701",        "label": 1, "skill_type": "cert"},
    {"taxonomy": "CISSP",                  "cv_phrase": "أعمل حالياً على التحضير لشهادة CISSP",           "label": 1, "skill_type": "cert"},

    # hard negatives
    {"taxonomy": "Docker",                 "cv_phrase": "عملت في الميناء أتولى رصف الحاويات على السفن",  "label": 0, "skill_type": "technical"},
    {"taxonomy": "Python",                 "cv_phrase": "بحثت في موائل ثعبان البايثون أثناء تدريب بيئي",  "label": 0, "skill_type": "technical"},
    {"taxonomy": "machine learning",       "cv_phrase": "شغّلت آلات ثقيلة في خط الإنتاج الصناعي",         "label": 0, "skill_type": "technical"},
    {"taxonomy": "scrum",                  "cv_phrase": "لعبت في فريق الرغبي الجامعي ضمن تشكيلة scrum",   "label": 0, "skill_type": "technical"},
    {"taxonomy": "AWS Certified Solutions Architect", "cv_phrase": "حضرت جلسة تعريفية بـ AWS في الجامعة",  "label": 0, "skill_type": "cert"},
    {"taxonomy": "PMP",                    "cv_phrase": "قرأت دليل PMBOK دون التسجيل في الاختبار",        "label": 0, "skill_type": "cert"},
    {"taxonomy": "CISSP",                  "cv_phrase": "أنوي التقدم لشهادة CISSP في غضون عام",           "label": 0, "skill_type": "cert"},
]


def paraphrase_pairs_ar() -> List[Dict[str, Any]]:
    return list(_PARAPHRASE_AR)
