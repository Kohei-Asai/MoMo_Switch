import classifier

test_x, test_y = classifier.make_tensors_from_mat(['data/data_5.mat'])[0]

model = classifier.load_model(
    model_path='model_9freedom.pth',
    input_dim=9,
    hidden_dim=128,
    target_dim=5
)

predicted_y = classifier.classificate(model, test_x, -0.15)

classifier.compare_graph(test_y, predicted_y)

print(classifier.index2category[0])