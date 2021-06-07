from matplotlib import image, pyplot as plt
import numpy  as np
from scipy import stats
from sklearn.metrics import accuracy_score
from tkinter import *
import  random
# read test1.bmp
test1_path = "G:\master_matus\99_2\PGM\HW1\\test1.bmp"
image_pixel = image.imread(test1_path)[:,:,0]
print(image_pixel.shape)
# get dimension of image.
row, col = image_pixel.shape[0], image_pixel.shape[1]
# get prior probability.
prior_probability = np.zeros(3)
labels_map = {
     0:0,
     1:127,
     2:255
}
l = {
     0:0,
     127:1,
     255:1
}
flat_image = np.ndarray.flatten(image_pixel.reshape(-1, col * row))
for i, p in enumerate(flat_image):
  prior_probability[l[p]] += 1
  flat_image[i] = l[p]
prior_probability /= col*row
print(prior_probability)


# add gaussian noise to image.
def add_noise(sigma,mean):
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy_image = image_pixel / 255 + gauss
    print(noisy_image)
    plt.imshow(noisy_image,cmap="gray")
    # image.imsave( "noise_image" + str(sigma)+".png", noisy_image)
    plt.show()
    return noisy_image

# maximum likelihood.
# label 0

sigma = 0.05
maximum_likelihood =[]
maximum_likelihood.append(stats.norm(0, sigma))
maximum_likelihood.append(stats.norm(float(127/255), sigma))
maximum_likelihood.append( stats.norm(1, sigma))


# segment image  by naive bayse.
def segmentation_naive_bys(noisy_image,name):
    predict_segment = []
    for i in range(row):
        for j in range(col):
            pixel = noisy_image[i, j]
            max_prob = 0
            s = -1
            for k, l in enumerate(maximum_likelihood):
                prob = maximum_likelihood[k].pdf(pixel[0]) + \
                       maximum_likelihood[k].pdf(pixel[1]) + \
                       maximum_likelihood[k].pdf(pixel[2])
                prob /= 3
                if max_prob < prob:
                    max_prob = prob
                    s = k
            predict_segment.append(s)
            noisy_image[i, j] = np.zeros(3) + labels_map[s] / 255
    plt.imshow(noisy_image)
    image.imsave(name,noisy_image)
    plt.show()
    print(accuracy_score(flat_image, predict_segment))


# defining of factor on label of neighbouring pixels.
def label_factor(label1, label2,beta):
    """
    :param label1: shows pixel is in what segmsent.
    :param label2: shows neighbouring pixel is in what segment.
    :param beta: is parameter that should tune it.
    :return: factor_value for these inputs(label1, label2)
    """
    factor_value = 0
    if label1 == label2 :
        factor_value = -beta
    else:
        factor_value = beta
    return factor_value


# get neighbour of pixel.
def get_neighbour(dimension, n, row, col):
    if dimension[0] == 0 and dimension[1] ==0:
        if n == 4:
            neighbour = [ [dimension[0] + 1, dimension[1]],
                          [dimension[0], dimension[1] + 1]]
        elif n == 8:
            neighbour = [ [dimension[0] + 1, dimension[1]],
                          [dimension[0], dimension[1] + 1],
                          [dimension[0] + 1, dimension[1] + 1]
                        ]
    elif dimension[1] == col - 1 and dimension[0] == 0:
        if n == 4:
            neighbour = [
                [dimension[0], dimension[1] - 1], [dimension[0]+1, dimension[1] ]]
        elif n == 8:
            neighbour = [[dimension[0] + 1, dimension[1]],
                         [dimension[0], dimension[1] - 1],
                         [dimension[0] + 1, dimension[1] - 1]]
    elif dimension[0] == row - 1 and dimension[1] == 0:
        if n == 4:
            neighbour = [[dimension[0] - 1, dimension[1]],
                         [dimension[0], dimension[1] + 1]]
        elif n == 8:
            neighbour = [[dimension[0] - 1, dimension[1]]
                , [dimension[0], dimension[1] + 1],
                         [dimension[0] - 1, dimension[1] + 1]]
    elif dimension[0] == 0:
        if n == 4:
            neighbour = [ [dimension[0] + 1, dimension[1]],
                         [dimension[0], dimension[1] - 1], [dimension[0], dimension[1] + 1]]
        elif n == 8:
            neighbour = [[dimension[0] + 1, dimension[1]],
                         [dimension[0], dimension[1] - 1], [dimension[0], dimension[1] + 1],
                         [dimension[0] + 1, dimension[1] + 1],
                         [dimension[0] + 1, dimension[1] - 1]]
    elif dimension[1] == 0:
        if n == 4:
            neighbour = [[dimension[0] - 1, dimension[1]], [dimension[0] + 1, dimension[1]],
                         [dimension[0], dimension[1] + 1]]
        elif n == 8:
            neighbour = [[dimension[0] - 1, dimension[1]], [dimension[0] + 1, dimension[1]],
                        [dimension[0], dimension[1] + 1],
                         [dimension[0] + 1, dimension[1] + 1],
                          [dimension[0] - 1, dimension[1] + 1]]
    elif dimension[0] == row -1 and dimension[1] == col -1 :
        if n == 4:
            neighbour = [[dimension[0] - 1, dimension[1]],
                         [dimension[0], dimension[1] - 1]]
        elif n == 8:
            neighbour = [[dimension[0] - 1, dimension[1]],
                         [dimension[0], dimension[1] - 1],
                         [dimension[0] - 1, dimension[1] - 1]
                       ]
    elif dimension[0] == row -1 :
        if n == 4:
            neighbour = [[dimension[0] - 1, dimension[1]],
                         [dimension[0], dimension[1] - 1], [dimension[0], dimension[1] + 1]]
        elif n == 8:
            neighbour = [[dimension[0] - 1, dimension[1]],
                         [dimension[0], dimension[1] - 1], [dimension[0], dimension[1] + 1],
                         [dimension[0] - 1, dimension[1] - 1],
                       [dimension[0] - 1, dimension[1] + 1]]
    elif  dimension[1] == col -1 :
        if n == 4:
            neighbour = [[dimension[0] - 1, dimension[1]], [dimension[0] + 1, dimension[1]],
                         [dimension[0], dimension[1] - 1]]
        elif n == 8:
            neighbour = [[dimension[0] - 1, dimension[1]], [dimension[0] + 1, dimension[1]],
                         [dimension[0], dimension[1] - 1],
                         [dimension[0] - 1, dimension[1] - 1],
                         [dimension[0] + 1, dimension[1] - 1]]


    else:
        if n == 4:
            neighbour = [[dimension[0] - 1, dimension[1]], [dimension[0] + 1, dimension[1]],
                         [dimension[0], dimension[1] - 1], [dimension[0], dimension[1] + 1]]
        elif n == 8:
            neighbour = [[dimension[0] - 1, dimension[1]], [dimension[0] + 1, dimension[1]],
                         [dimension[0], dimension[1] - 1], [dimension[0], dimension[1] + 1],
                         [dimension[0] - 1, dimension[1] - 1], [dimension[0] + 1, dimension[1] + 1],
                         [dimension[0] + 1, dimension[1] - 1], [dimension[0] - 1, dimension[1] + 1]]

    return neighbour


def energy_function(feature_array, label_array,gaussian_param,n,beta):
    """

    :param feature_array: features value of input image.
    :param label_array: selected labels for pixels in current state.
    :param n: The number of neighbors to be considered.
    :return: value of energy function for current state.
    """
    row = feature_array.shape[0]
    col = feature_array.shape[1]
    a = 0
    b = 0
    energy_value = 0
    for i in range(row):
        for j in range(col):
            a += np.sqrt(2*np.pi*gaussian_param[label_array[i,j]]['sigma'])+((feature_array[i,j] - gaussian_param[label_array[i,j]]['mean'])**2) \
                 / (2 * np.power(gaussian_param[label_array[i,j]]['sigma'],2))
            for dim in get_neighbour([i,j], n, row, col):

                b += label_factor(label_array[i,j], label_array[dim[0],dim[1]], beta)

    energy_value = a + b
    return energy_value


def next_labeling(label_array, energy, gaussian_param,feature_array,beta,n):
    new_label = label_array.copy()
    rand_pixel = np.random.randint(0,label_array.shape[0]*label_array.shape[1])
    rand_dim = np.unravel_index(rand_pixel, shape=(label_array.shape[0],label_array.shape[1]))
    current_label = label_array[rand_dim[0], rand_dim[1]]
    a = np.sqrt(2 * np.pi * gaussian_param[label_array[rand_dim[0], rand_dim[1]]]['sigma']) + (
                (feature_array[rand_dim[0], rand_dim[1]] - gaussian_param[label_array[rand_dim[0], rand_dim[1]]]['mean']) ** 2) \
         / (2 * np.power(gaussian_param[label_array[rand_dim[0], rand_dim[1]]]['sigma'], 2))
    b = 0
    for dim in get_neighbour([rand_dim[0], rand_dim[1]], n, row, col):
        b += label_factor(label_array[rand_dim[0], rand_dim[1]], label_array[dim[0],dim[1]], beta)
    energy -= (a+b)
    new= random.choice([i for i in range(0, 3) if i != current_label])
    a = np.sqrt(2 * np.pi * gaussian_param[new]['sigma']) + (
            (feature_array[rand_dim[0], rand_dim[1]] - gaussian_param[new][
                'mean']) ** 2) \
        / (2 * np.power(gaussian_param[new]['sigma'], 2))
    b = 0
    for dim in get_neighbour([rand_dim[0], rand_dim[1]], n, row, col):
        b += label_factor(new, label_array[dim[0],dim[1]], beta)
    energy += (a+b)
    new_label[rand_dim[0], rand_dim[1]] = new
    return new_label,energy


def random_initializing_labels(row, col):
    label_array = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            label_array[i,j] = np.random.randint(0,3)
    return label_array


def simulated_annealing(feature_array, gaussian_param, n, beta, T):
    label_array = random_initializing_labels(feature_array.shape[0], feature_array.shape[1])
    T0 = T
    iter = 0
    E_current = energy_function(feature_array, label_array, gaussian_param, n, beta)
    while True:
      next_labels,  E_next = next_labeling(label_array,E_current,gaussian_param,feature_array,beta,n)
      delta_E = E_next - E_current
      if delta_E < 0:
          label_array = next_labels
          E_current = E_next
      elif delta_E > 0 :
          alpha = random.uniform(0,1)
          if alpha < np.exp(-delta_E/T):
              label_array = next_labels
              E_current = E_next
      T = (1 - iter/100000000) * T0
      if T < 0 :
          T = 0
      print(iter,T, E_current, E_next)
      if iter >100000000:
          break
      iter += 1
      E_current = E_next
    segment_image = np.zeros((feature_array.shape[0],feature_array.shape[1]))
    for i in range(feature_array.shape[0]):
        for j in range(feature_array.shape[1]):
            segment_image[i,j] = labels_map[label_array[i,j]]
    plt.imshow(segment_image/255, cmap="gray")
    plt.show()


# use markov random field for segmentation.
def MRF_segmentation():
    guassian_param = {
        0: {
            "sigma":0.1,
            "mean":0
        },
        1: {
            "sigma": 0.1,
            "mean": 0.5
        },
        2: {
            "sigma":0.1,
            "mean":1
        }
    }
    simulated_annealing(add_noise(0.05, 0),guassian_param, 4, 5, 5)


if __name__ == "__main__":
    MRF_segmentation()
# part1:a
#     noisy_image1 = add_noise(0.1, 0)
#     segmentation_naive_bys(noisy_image1, "seg_1.png")
#     noisy_image2 = add_noise(0.05, 0)
#     segmentation_naive_bys(noisy_image2, "seg_2.png")
#     noisy_image3 = add_noise(0.3, 0)
#     segmentation_naive_bys(noisy_image3, "seg_3.png")
#     top = Tk()
#     mb =  Menubutton ( top, text = "algorithm")
#     mb.grid()
#     mb.menu  =  Menu ( mb, tearoff = 0 )
#     mb["menu"]  =  mb.menu
#     cVar  = IntVar()
#     aVar = IntVar()
#     mb.menu.add_checkbutton ( label ='Naive bayse', variable = cVar )
#     mb.menu.add_checkbutton ( label = 'MRF', variable = aVar )
#     mb.e
#     mb.pack()
#     top.mainloop()

