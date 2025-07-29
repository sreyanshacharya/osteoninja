# OsteoNinja  
> Deep learning meets bone-breaking brilliance.

OsteoNinja is a two-part X-ray fracture classification project built to level up my Deep Learning game. Starting with a CNN from scratch and leveling up to transfer learning with ResNet50, this project explores the challenges of training on small datasets, avoiding overfitting, and still scoring a decent shot at real-world performance.  

It ain’t built for medicos — it’s built for learning.

---

## Models

### v1 – CNN From Scratch  
- Built & trained on augmented dataset (~4000+ images per class).  
- Performs well in validation but fails in real-world unseen images (classic case of overfitting and using over-augmented data).

### v2 – Transfer Learning w/ ResNet50  
- Trained on the original non-augmented dataset (420 images in total) using ResNet50 as base.  
- Smart unfreezing + light tuning resulted in solid performance.  
- Holds up much better in real-world tests compared to v1.  

---

## Real World Evaluation

- After training, the models were tested on a tiny, unseen real-world dataset to see how they'd generalize.
- Spoiler: v2 > v1. Every time.
- On average, v1 got only 1-2 out of 4 x-rays correct, while v2 consistently predicted 3 x-rays or higher out of 4 correctly.
- You can test this yourself with the x-rays provided in the [realimages](./realimages) subfolder! Scroll below to see how you can run it.
---

## Tech Stack

- Python 
- TensorFlow / Keras  
- Matplotlib, NumPy  
- Good ol’ trial and error

---

## Dataset

Used this [Kaggle X-ray fracture dataset](https://www.kaggle.com/datasets/foyez767/x-ray-images-of-fractured-and-healthy-bones).  
Downloaded manually — no API wizardry. Props to the author.

---

## Notes

- **This project is not meant for clinical use.** It's for practice, understanding nuances of small datasets and the impact of transfer learning.
- Dropout, Image Augmentation, Learning Rate Tuning, and Layer Unfreezing all played major roles in the final accuracy.
- The final version isn’t perfect, but it’s the best this dataset can offer without diving into med school.

---

## Run it Yourself

- Refer the [runitlocally.md](./runitlocally.md) file for info.

## Author 

- **Sreyansh Acharya** - CS Student at GITAM Hyd and aspiring ML Engineer with interests in Deep Learning, Astronomy and Rock Music 🤘🏻
- Connect with me:
    - sreyanshacharyaa@gmail.com
    - [github.com/sreyanshacharya](https://github.com/sreyanshacharya)
