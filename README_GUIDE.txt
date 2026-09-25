================================================================================
KAISI TAIYARI KARNI HAI (STEP-BY-STEP EXAM GUIDE)
================================================================================

[STEP 1: AAJ RAAT GHAR PAR (1 MINUTE)]
1. Google Drive kholo (drive.google.com).
2. Is folder (ULT_Drive_Upload) ki saari files (pr1.py se pr11.py aur ult.py) 
   apne Google Drive ke Main page (MyDrive) par drag & drop karke upload kar do.

--------------------------------------------------------------------------------

[STEP 2: KAL EXAM ME COLAB KE ANDAR (NO TABS NEEDED!)]
Jab examiner screen par bole "Write Practical 4" (ya koi bhi practical):

1. Pehle cell me Drive mount karo (Sirf ek baar):
   from google.colab import drive
   drive.mount('/content/drive')

2. Ab jo practical likhna ho, agle cell me bas ye line likho:
   %load /content/drive/MyDrive/pr4.py

   -> Aur Shift + Enter dabao!
   -> JADU: Colab us line ko hata kar Pura Python Code cell me paste kar dega!
   -> Upar wali comment line (# %load ...) hata do aur Run kar do.

--------------------------------------------------------------------------------

[STEP 3: EXAM KHATAM HONE KE BAAD (SAB KUCH DELETE / HIDE KARNA)]
1. Jis cell me aapne drive mount kiya tha, us cell ko select karo aur 
   keyboard par dabao: Ctrl + M + D (Cell delete ho jayega).

2. Ek khali cell me ye run karo taaki Drive disconnect ho jaye:
   from google.colab import drive
   drive.flush_and_unmount()

3. Is cell ko bhi Ctrl + M + D karke delete kar do.

Examiner ko kuch pata nahi chalega, aur zero trace rahega!
================================================================================
