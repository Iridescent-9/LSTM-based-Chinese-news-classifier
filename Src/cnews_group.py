import os
from tqdm import tqdm


def _read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        temp = f.read().replace('\n', '').replace('\t', '').replace('\u3000', '')
        f.close()
    return temp


def save_file(dirname):
    train_data = []
    test_data = []
    val_data = []
    for category in os.listdir(dirname):
        cat_dir = os.path.join(dirname, category)
        if not os.path.isdir(cat_dir):
            continue

        print(f"Processing category {category}...")
        files = [os.path.join(cat_dir, cur_file) for cur_file in os.listdir(cat_dir)]
        file_list = os.listdir(cat_dir)
        total_files = len(file_list)

        train_threshold = int(0.6 * total_files)
        test_threshold = int(0.8 * total_files)
        for count, cur_file in enumerate(tqdm(files)):
            content = _read_file(cur_file)
            if count < train_threshold:
                train_data.append((category, content))
            elif count < test_threshold:
                test_data.append((category, content))
            else:
                val_data.append((category, content))

        print('Finished:', category)

    # Write data to files
    with open("D:\MyCode\\Graduation project\\Data\\cnews.train.txt", 'w', encoding='utf-8') as f_train, open(
            "D:\MyCode\\Graduation project\\Data\\cnews.test.txt", 'w', encoding='utf-8') as f_test, open(
        "D:\MyCode\\Graduation project\\Data\\cnews.val.txt", 'w', encoding='utf-8') as f_val:
        for category, content in train_data:
            f_train.write(category + '\t' + content + '\n')
        for category, content in test_data:
            f_test.write(category + '\t' + content + '\n')
        for category, content in val_data:
            f_val.write(category + '\t' + content + '\n')

    f_train.close()
    f_test.close()
    f_val.close()


if __name__ == '__main__':
    save_file("D:\\MyCode\\Graduation project\\THUCNews\\THUCNews")
