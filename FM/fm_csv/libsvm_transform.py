def load_featindex():
    featindex = {}
    with open("./churn_featindex.txt", encoding="utf8") as f:
        for line in f.readlines():
            index_value = line.strip()
            if index_value:
                featindex[index_value.split()[0]] = index_value.split()[1]
    return featindex


def svmlight_transform(svm_file, csv_file, featindex):
    with open(svm_file, "w", encoding="utf8") as f1:
        with open(csv_file, "r", encoding="utf8") as f2:
            next(f2)
            for line in f2.readlines():
                feature = line.strip().split(",")[1:-1]
                label = line.strip().split(",")[-1]
                libsvm_feature = []
                for index, value in enumerate(feature):
                    index_value = str(index) + ":" + value
                    location = featindex[index_value]
                    libsvm_feature.append(location + ":" + "1")
                libsvm = label + " " + " ".join(libsvm_feature)
                f1.write(libsvm + "\n")


if __name__ == "__main__":
    featindex = load_featindex()
    svmlight_transform("./churn_train.svm", "./churn_train.csv", featindex)
    svmlight_transform("./churn_test.svm", "./churn_test.csv", featindex)

