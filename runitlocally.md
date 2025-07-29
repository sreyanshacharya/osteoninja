# Run It Locally

Wanna test OsteoNinja on your own machine? Here's the breakdown.

---

## Requirements

Make sure you’ve got the basics ready:

- Python 3.8+
- TensorFlow (preferably 2.12+)
- numpy
- matplotlib
- tqdm
- scikit-learn
- (Optional) GPU acceleration for faster training

Install everything with:
- pip install -r requirements.txt

## Project Structure

  - osteoninja/
  - |── augdata/ # Heavily augmented dataset (v1)
  - │ ├── Fractured/
  - │ └── Non-Fractured/
  - |
  - ├── ogdata/ # Original dataset (v2)
  - │ ├── Fractured/
  - │ └── Non-Fractured/
  - |
  - ├── realworldimages/ # Unseen test images (IRL X-rays)
  - |
  - ├── v1/
  - │ ├── train.py
  - │ └── test.py
  - |
  - ├── v2/
  - │ ├── train.py
  - │ └── test.py
  - └── README.md

- **Make sure the structure is as mentioned above!**

## Run It
### v1 - CNN built from scratch
  - Ensure you are currently active in "v1" folder.
  - Run "train.py" to initialise model and train it.
  - Ensure your test x-rays are in the correct path ([project]/realworldimages/)
  - Run "test.py" to run the model on the test x-rays.

### v2 - Transferred Model w/ ResNet50
  - Ensure you are currently active in "v2" folder.
  - Run "train.py" to initialise model and train it.
  - Ensure your test x-rays are in the correct path ([project]/realworldimages/)
  - Run "test.py" to run the model on the test x-rays.
