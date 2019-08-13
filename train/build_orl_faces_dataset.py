import random
import glob
import pickle

if __name__ == '__main__':
    faces = glob.glob('../data/faces/orl_faces/*')
    train = []
    test = []

    print('splitting data set and saving to disk...')
    for face in faces:
        examples = glob.glob(face + '/*')
        # split into train and test set
        random.seed(230)
        examples.sort()
        random.shuffle(examples)
        split = int(0.7 * len(examples))
        train += examples[:split]
        test += examples[split:]

    splits = {'train': train, 'test': test}

    for split, filenames in splits.items():
        pkl_file = '../data/faces/orl_faces/' + split
        with open(pkl_file, 'wb') as f:
            pickle.dump(filenames, f)


