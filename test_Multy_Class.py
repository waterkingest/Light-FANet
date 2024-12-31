
import os, time
from operator import add
import json
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
import torch.nn as nn
from model import FANet,NewFANet,l3_FANet
from utils import create_dir, seeding, init_mask, rle_encode, rle_decode, load_data
from collections import defaultdict
from Miou import fast_hist,per_class_iu,per_class_PA

def comput_miou(y_true, y_pred):
    hist = fast_hist(y_true, y_pred, 2)
    # print(hist)
    mIoUs=per_class_iu(hist)
    miou=round(np.nanmean(mIoUs) , 2)
    # print(miou)
    return miou

def precision_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection) / (y_pred.sum() + 1e-15)

def recall_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)

def F2_score(y_true, y_pred, beta=2):
    p = precision_score(y_true,y_pred)
    r = recall_score(y_true, y_pred)
    return (1+beta**2.) *(p*r) / float(beta**2*p + r + 1e-15)

def dice_score(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def jac_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

def calculate_metrics(y_true, y_pred, img):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    y_pred = y_pred > 0.5
    y_pred = y_pred.reshape(-1)
    y_pred = y_pred.astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.reshape(-1)
    y_true = y_true.astype(np.uint8)

    ## Score
    score_jaccard = jac_score(y_true, y_pred)
    score_f1 = dice_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_fbeta = F2_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)
    score_miou=comput_miou(y_true, y_pred)
    # print(score_miou)
    confusion = confusion_matrix(y_true, y_pred)
    if confusion.shape[1] > 1:
      if float(confusion[0,0] + confusion[0,1]) != 0:
          score_specificity = float(confusion[0,0]) / float(confusion[0,0] + confusion[0,1])
      else:
          score_specificity = 0.0
    else:
      score_specificity = 0.0
    return [score_jaccard, score_f1, score_recall, score_precision, score_specificity, score_acc, score_fbeta , score_miou]

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """
    def gather(self, outputs, output_device):
        # Note that I don't actually want to convert everything to the output_device
        return sum(outputs, [])


def convert_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
def find_max(dd):
    matrices = [dd[key] for key in dd.keys()]
    stacked_matrices = np.stack(matrices)
    max_values = np.max(stacked_matrices, axis=0)
    max_index=np.argmax(stacked_matrices,axis=0)
    # np.savetxt('max_index.csv', max_index, delimiter=',')
    keys = list(dd.keys())
    result_matrix=[]
    for xx,row in enumerate(max_values):
        result_matrix.append([])
        for yy, max_val in enumerate(row):
            result_matrix[xx].append((keys[max_index[xx][yy]],max_val))
    return result_matrix
if __name__ == "__main__":
    color_dict={'car':(255,0,0),
                'people':(0,0,255),
                'path':(128,64,128)
                }
    """ Seeding """
    seeding(42)
    
    """ Load dataset """
    path = "VOCdevkit"
    (train_x, train_y), (test_x, test_y) = load_data(path)
    
    test_num=50
    test_x,test_y=test_x[:test_num],test_y[:test_num]
    """ Hyperparameters """
    size = (512, 512)
    num_iter = 15
    # checkpoint_dict={
    #     'car':r'files\checkpoint_car_original.pth',
    #     'people':r'files\people_original.pth',
    #     'path':r'files\path_original.pth',
    # }
    
    # checkpoint_dict={
    #     'car':r'files\3layer_car.pth',
    #     'people':r'files\3layer_people.pth',
    #     'path':r'files\3layer_road.pth',
    # }
    
    checkpoint_dict={
        'car':r'files\checkpoint_att3layercar.pth',
        'people':r'files\checkpoint_att3ped.pth',
        'path':r'files\checkpoint_a3road.pth',
    }
    """ Directories """
    dir_name='3layer_attention'
    create_dir(dir_name)

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    # model = model.to(device)
    
    model_dict={}
    for key,value in checkpoint_dict.items():
        # model = FANet()
        model=l3_FANet()
        # model=NewFANet()
        temp_model=nn.DataParallel(model)
        temp_model.load_state_dict(torch.load(value, map_location=device),strict=False)
        model_dict[key]=CustomDataParallel(temp_model).to(device)
        model_dict[key].eval()
        
        
#     model = nn.DataParallel(model)
#     model.load_state_dict(torch.load(checkpoint_path, map_location=device),strict=False)
#     model = CustomDataParallel(model).to(device)
# #     model=model.to(device)
#     model.eval()

    """ Testing """
    mask_dict=defaultdict(list)
    for key,model in model_dict.items():
        prev_masks = init_mask(test_x, size)
        save_data = []
        file = open(f"{dir_name}/test_results.csv", "a+")
        file_path=f"{dir_name}/test_results.csv"
        is_empty = os.stat(file_path).st_size == 0
        if is_empty:
            file.write("Category,Iteration,Jaccard,F1,Recall,Precision,Specificity,Accuracy,F2,Mean Time,Mean FPS,mIoU\n")

        for iter in range(num_iter):

            metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            tmp_masks = []
            time_taken = []

            for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
                ## Image
                image = cv2.imread(x, cv2.IMREAD_COLOR)
                image = cv2.resize(image, size)
                img_x = image
                image = np.transpose(image, (2, 0, 1))
                image = image/255.0
                image = np.expand_dims(image, axis=0)
                image = image.astype(np.float32)
                image = torch.from_numpy(image)
                image = image.to(device)

                ## Mask
                mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, size)
                mask = np.expand_dims(mask, axis=0)
                mask = mask/255.0
                mask = np.expand_dims(mask, axis=0)
                mask = mask.astype(np.float32)
                mask = torch.from_numpy(mask)
                mask = mask.to(device)

                ## Prev mask
                pmask = prev_masks[i]
                pmask = " ".join(str(d) for d in pmask)
                pmask = str(pmask)
                pmask = rle_decode(pmask, size)
                pmask = np.expand_dims(pmask, axis=0)
                pmask = np.expand_dims(pmask, axis=0)
                pmask = pmask.astype(np.float32)
                if iter == 0:
                    pmask = np.transpose(pmask, (0, 1, 3, 2))
                pmask = torch.from_numpy(pmask)
                pmask = pmask.to(device)

                with torch.no_grad():
                    """ FPS Calculation """
                    start_time = time.time()
                    network_out=model([image, pmask])
                    pred_y = torch.sigmoid(network_out)
                    end_time = time.time() - start_time
                    time_taken.append(end_time)

                    score = calculate_metrics(mask, pred_y, img_x)
                    metrics_score = list(map(add, metrics_score, score))
                    pred_y = pred_y[0][0].cpu().numpy()
                    if iter==num_iter-1:
                        filtered_array = np.where(pred_y > 0.5, pred_y, 0)
                        mask_dict[key].append(filtered_array)
                        
                    pred_y = pred_y > 0.5
                    # print(pred_y)
                    pred_y = np.transpose(pred_y, (1, 0))
                    pred_y = np.array(pred_y, dtype=np.uint8)
                    pred_y = rle_encode(pred_y)
                    prev_masks[i] = pred_y
                    tmp_masks.append(pred_y)

            """ Mean Metrics Score """
            print(metrics_score)
            jaccard = metrics_score[0]/len(test_x)
            f1 = metrics_score[1]/len(test_x)
            recall = metrics_score[2]/len(test_x)
            precision = metrics_score[3]/len(test_x)
            specificity = metrics_score[4]/len(test_x)
            acc = metrics_score[5]/len(test_x)
            f2 = metrics_score[6]/len(test_x)
            miou=metrics_score[7]/len(test_x)

            """ Mean Time Calculation """
            mean_time_taken = np.mean(time_taken)
            print("Mean Time Taken: ", mean_time_taken)
            mean_fps = 1/mean_time_taken

            print(f"Category: {key} ,Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Specificity: {specificity:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f} - Mean Time: {mean_time_taken:1.7f} - Mean FPS: {mean_fps:1.7f} - mIoU: {miou:1.7f}")

            save_str = f"{key},{iter+1},{jaccard:1.4f},{f1:1.4f},{recall:1.4f},{precision:1.4f},{specificity:1.4f},{acc:1.7f},{f2:1.7f},{mean_time_taken:1.7f},{mean_fps:1.7f},{miou:1.7f}\n"
            file.write(save_str)

            save_data.append(tmp_masks)
        save_data = np.array(save_data, dtype=object)
        # """ Saving the masks. """
        # create_dir(key)
        # for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        #     image = cv2.imread(x, cv2.IMREAD_COLOR)
        #     image = cv2.resize(image, size)

        #     mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        #     mask = cv2.resize(mask, size)
        #     # mask = mask / 255
        #     # mask = (mask > 0.5) * 255
        #     mask = mask_parse(mask)

        #     name = y.split("\\")[-1].split(".")[0]
        #     sep_line = np.ones((size[0], 10, 3)) * 128
        #     tmp = [image, sep_line, mask]

        #     for data in save_data:
        #         tmp.append(sep_line)
        #         d = data[i]
        #         d = " ".join(str(z) for z in d)
        #         d = str(d)
        #         d = rle_decode(d, size)
        #         d = d * 255
        #         d = mask_parse(d)

        #         tmp.append(d)

        #     cat_images = np.concatenate(tmp, axis=1)
        #     # print(name)
        #     cv2.imwrite(f"{key}/{name}.png", cat_images)
    """ Saving the masks. """
    category_list=list(mask_dict.keys())
    print(category_list)
    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)


        temp_mask_dict={}
        for key in category_list:
            temp_mask_dict[key]=mask_dict[key][i]
        mix_matrix=find_max(temp_mask_dict)

        name = y.split("\\")[-1].split(".")[0]
        sep_line = np.ones((size[0], 10, 3)) * 128
        tmp = [image, sep_line]
        mask=np.zeros((size[0],size[1], 3), dtype=np.uint8)
        
        
        for ii in range(len(mix_matrix)):
            for jj in range(len(mix_matrix[ii])):
                category,value=mix_matrix[ii][jj]
                if value>0.5:
                    mask[ii][jj]=color_dict[category]
                    
        # cv2.imwrite(f"multi_results/{name}_mask.png", mask)             
        # for key in category_list:
        #     for ii in range(len(mask_dict[key][i])):
        #         for jj in range(len(mask_dict[key][i][ii])):
        #             value=mask_dict[key][i][ii][jj]
        #             if value>0.5:
        #                 mask[ii][jj]=color_dict[key]
        tmp.append(mask)


        cat_images = np.concatenate(tmp, axis=1)
        # print(name)
        cv2.imwrite(f"{dir_name}/{name}.png", cat_images)
        # print('yes')
