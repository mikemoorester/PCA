#
# Convert the mnsit format to a csv format
#
# original data can be obtained from: http://yann.lecun.com/exdb/mnist/
#
# Use this program to onvert the idx3 format to csv so that it easy to 
# import into pandas
#
# NOTE to self: on my laptop I've stored the data in /data/ml/mnist/
#
def write_csv_header(o):
    o.write("label")
    for i in range(0,784):
        o.write(","+str(i))
    o.write("\n")

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []
    write_csv_header(o)

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

convert("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "mnist_train.csv", 60000)
convert("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", "mnist_test.csv", 10000)
