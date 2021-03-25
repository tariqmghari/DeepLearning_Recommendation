import torch
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def modelTrain(model, trainLoader, optimizer, lossFnc, epoch=5):
    l = len(trainLoader) 
    print('--- Training started ---')
    for epoch in range(epoch):
        running_loss = 0.0
        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = lossFnc(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i+1) % l == 0:
                print('epoch: %d loss: %.4f' %(epoch + 1, running_loss / l))
    print('--- Finished Training ---')

def modelAccurcy(model, testLoader, classes=None):
    correct = 0
    total = 0
    if classes is not None:
        class_correct = list(0. for i in range(len(classes)))
        class_total = list(0. for i in range(len(classes)))

    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if classes is not None:
                c = (predicted == labels)
                for i in range(len(c)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

    if classes is not None:  
        print("----- Accuracy for each class ----")             
        for i in range(len(classes)):
            if class_total[i] != 0:
                print('- %5s : %.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
            else:
                print(classes[i],"doesn't exist in DataTest")
        print("-"*50)

    print('Accuracy of the network on the test images: %.2f %%' % (100 * correct / total))

def saveModel(model, Path):
    torch.save(model.state_dict(), Path)
    return "Model saved successfully"

def loadModel(model, path):
    model.load_state_dict(torch.load(path))
    return "Model loaded successfully"