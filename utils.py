from lib import *
from config import *


def make_datapath_list(phase="train"):
    rootpath = "hymenoptera_data\\"
    target_path = osp.join(rootpath + phase + "/**/*.jpg")

    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list


def train_model(net, dataloader_dict, criterior, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))

        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0      #tổng độ lỗi trong một epoch
            epoch_corrects = 0    #số lượng dự đoán đúng trong một epoch

            if (epoch == 0) and (phase == "train"):
                continue
            for inputs, labels in dataloader_dict[phase]:

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)       #[batch_size, num_classes]
                    loss = criterior(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_accuracy = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_accuracy))
    torch.save(net.state_dict(), save_path)

def params_to_update(net):
    params_to_update_1 = []
    params_to_update_2 = []
    params_to_update_3 = []

    update_param_name_1 = ["features"]
    update_param_name_2 = ["classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
    update_param_name_3 = ["classifier.6.weight", "classifier.6.bias"]

    for name, param in net.named_parameters():
        if name in update_param_name_1:
            param.requires_grad = True
            params_to_update_1.append(param)
        elif name in update_param_name_2:
            param.requires_grad = True
            params_to_update_2.append(param)
        elif name in update_param_name_3:
            param.requires_grad = True
            params_to_update_3.append(param)

        else:
            param.requires_grad = False
    return params_to_update_1, params_to_update_2, params_to_update_3


def load_model(net, model_path):
    load_weights = torch.load(model_path)
    net.load_state_dict(load_weights)

    # print(net)
    # for name, param in net.named_parameters():
    #     print(name, param)

    return net