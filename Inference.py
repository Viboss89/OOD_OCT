import torch
import tqdm
from torch.utils.data import DataLoader
from utils.config import DefaultConfig
from models.net_builder import net_builder
from dataprepare.dataloader import DatasetCFP, class_sampler, Folders_dataset
from torch.nn import functional as F
import matplotlib.pyplot as plt

import csv

def val(val_dataloader, model, args, Normal_average=0.0,device=None):

    print('\n')
    model.eval()
    tbar = tqdm.tqdm(val_dataloader, desc='\r')
    All_Infor = []

    with torch.no_grad():
        for i, img_data_list in enumerate(tbar):

            plt.imshow(img_data_list[0][0].permute(2,1,0))
            plt.show()
            Fundus_img = img_data_list[0].to(device)
            image_files = img_data_list[1]
            pred = model.forward(Fundus_img)
            evidences = [F.softplus(pred)]

            alpha = dict()
            alpha[0] = evidences[0] + 1

            S = torch.sum(alpha[0], dim=1, keepdim=True)
            E = alpha[0] - 1
            b = E / (S.expand(E.shape))

            u = args.num_classes / S

            batch = b.shape[0]

            pred_con = pred.argmax(dim=-1)



            for idx_bs_u in range(batch):

                if u[idx_bs_u] >= Normal_average:
                    """
                    All_Infor.append([
                        '/'.join(image_files[idx_bs_u].split('/')[-2:]),
                        pred_con.cpu().detach().float().numpy()[idx_bs_u],
                        "Unreliable"
                    ]
                    )
                else:
                    All_Infor.append([
                        '/'.join(image_files[idx_bs_u].split('/')[-2:]),
                        pred_con.cpu().detach().float().numpy()[idx_bs_u],
                        "Reliable"
                    ]
                    )
                    """
                    All_Infor.append([str(image_files[idx_bs_u].cpu().detach().int().numpy()), str(pred_con.cpu().detach().int().numpy()[idx_bs_u]), "Unreliable"])

                else: 
                    All_Infor.append([str(image_files[idx_bs_u].cpu().detach().int().numpy()), str(pred_con.cpu().detach().int().numpy()[idx_bs_u]),"Reliable"])


    return All_Infor


def main(args=None):


    args.net_work = "ResUnNet50"
    args.trained_model_path = './Trained/Model_Kellman/model_Test_011_Val_0.996431_0.971238_Test_0.995923_0.974201.pth.tar'
    # bulid model
    device = torch.device('cuda:{}'.format(args.cuda))
    args.device = device

    model = net_builder(args.net_work, args.num_classes).to(device)

    print('Model have been loaded!,you chose the ' + args.net_work + '!')
    # load trained model for test
    print("=> loading trained model '{}'".format(args.trained_model_path))
    checkpoint = torch.load(
        args.trained_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print('Done!')


    Thres = 0.0508
    Results_Heads = ["Imagefiles", 'Prediction results','Reliability']



    args.root = "../data"
    csv_file = "./Datasets/Pred_test_standard.csv"

    torch.manual_seed(0)
    split_ratio = 0.8
    data = Folders_dataset(path=args.root)
    train_data, test_data = torch.utils.data.random_split(data, [int(len(data)*split_ratio), len(data)-int(len(data)*split_ratio)])
    #test_data, val_data = torch.utils.data.random_split(val_data, [int(len(val_data)*0.2), len(val_data)-int(len(val_data)*0.2)])
    #train_sampler = class_sampler(data, train_data)
    #del train_data, val_data, test_data, data
    train_loader = DataLoader(train_data,
        batch_size=args.batch_size, pin_memory=True)
    val_loader = DataLoader(test_data,
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data,
        batch_size=args.batch_size, shuffle=True, pin_memory=True)

    Results_Contents = val(test_loader, model, args,
        Normal_average=Thres, device=device)
    with open(
            "Results_ood_OCTID.csv", 'a',     #modify name to generate reports with csv format
            newline='') as f:
        writer = csv.writer(f)
        writer.writerow(Results_Heads)
        writer.writerows(
            Results_Contents
        )


if __name__ == '__main__':
    args = DefaultConfig()

    main(args=args)