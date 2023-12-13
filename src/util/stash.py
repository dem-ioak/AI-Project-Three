# Stashed Code
# To reduce clutter in LogisticRegression class, I'm putting functions we currently have no need for here

def preprocess_data(image_data, label_data):
    """Preprocess and encode image and label data for model training"""
    image_data = one_hot_encode(image_data)
    label_data = np.array(label_data)
    flattened_data = image_data.reshape(image_data.shape[0], -1)
    flattened_data = np.c_[np.ones(len(flattened_data)), flattened_data] 
    return flattened_data, label_data


    
# Gradient Descent
def gd(self):
    """Run Gradient Descent to find `parameters` to minimize loss"""
    # Shuffle data before each epoch
    # random.shuffle(self.examples)
    # for i in range(len(self.examples)):
    #errors = self.loss(self.labels, self.predict())
    residuals = self.predict() - self.y_train
    gradient = np.dot(self.X_train.T, residuals)
    self.weights -= self.lr * gradient


def train_deterministic(self, epochs):
    """Run GD until # of epochs is exceeded OR convergence"""
    prev = float('inf')
    for epoch in range(epochs):
        self.gd()
        train_loss = self.loss(self.y_train, self.predict())
        if epoch % 5 == 0:
            print(f"{epoch} - Loss: {train_loss}")
            
        if prev - train_loss < self.epsilon:
            print(f"Stopping early at epoch {epoch} - Loss: {train_loss}")
            break
        prev = train_loss
        
    print(f"{epoch} - Loss: {train_loss}")