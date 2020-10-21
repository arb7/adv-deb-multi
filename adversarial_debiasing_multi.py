import sys
import numpy as np
import tensorflow as tf


"""
Adversarial debiasing implementation, compatible with multiclass classification,
continuous protected attribute, and generic pandas DataFrame input

This code is modified from IBM's AIF350 code at
https://github.com/IBM/AIF360/blob/master/aif360/algorithms/inprocessing/adversarial_debiasing.py
"""


class AdversarialDebiasingMulti():
    """Adversarial debiasing is an in-processing technique that learns a
    classifier to maximize prediction accuracy and simultaneously reduce an
    adversary's ability to determine the protected attribute from the
    predictions [5]_. This approach leads to a fair classifier as the
    predictions cannot carry any group discrimination information that the
    adversary can exploit.
    References:
        .. [5] B. H. Zhang, B. Lemoine, and M. Mitchell, "Mitigating Unwanted
           Biases with Adversarial Learning," AAAI/ACM Conference on Artificial
           Intelligence, Ethics, and Society, 2018.
    """

    def __init__(self,
                 protected_attribute_name,
                 num_labels,
                 scope_name,
                 sess,
                 seed=None,
                 adversary_loss_weight=0.1,
                 num_epochs=50,
                 batch_size=128,
                 classifier_num_hidden_units_1=100,
                 classifier_num_hidden_units_2=100,
                 adversary_num_hidden_units=100,
                 debias=True,
                 verbose=True,
                 fairness_def='parity',
                 saved_model=None):

        self.scope_name = scope_name
        self.seed = seed

        self.protected_attribute_name = protected_attribute_name
        self.num_labels = num_labels

        self.sess = sess
        self.adversary_loss_weight = adversary_loss_weight
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.classifier_num_hidden_units_1 = classifier_num_hidden_units_1
        self.classifier_num_hidden_units_2 = classifier_num_hidden_units_2
        self.adversary_num_hidden_units = adversary_num_hidden_units
        self.debias = debias
        self.verbose = verbose
        assert fairness_def in ['parity', 'equal_odds'], \
            "fairness_def must be one of: 'parity', 'equal_odds'"
        self.fairness_def = fairness_def

        self.features_dim = None
        self.features_ph = None
        self.protected_attributes_ph = None
        self.true_labels_ph = None
        self.pred_labels = None

        self.label_translate = {}

        self.saved_model = saved_model

    def _classifier_model(self, features, features_dim, keep_prob):
        """Compute the classifier predictions for the outcome variable.
        """

        with tf.compat.v1.variable_scope("classifier_model"):
            W1 = tf.compat.v1.get_variable('W1', [features_dim, self.classifier_num_hidden_units_1],
                                  initializer=tf.keras.initializers.GlorotUniform(seed=self.seed1))
            b1 = tf.Variable(tf.zeros(shape=[self.classifier_num_hidden_units_1]), name='b1')

            h1 = tf.nn.relu(tf.matmul(features, W1) + b1)
            h1 = tf.nn.dropout(h1, rate=1-keep_prob, seed=self.seed2)

            # BEGIN NEW

            W3 = tf.compat.v1.get_variable('W3', [self.classifier_num_hidden_units_1, self.classifier_num_hidden_units_2],
                                  initializer=tf.keras.initializers.GlorotUniform(seed=self.seed5))
            b3 = tf.Variable(tf.zeros(shape=[self.classifier_num_hidden_units_2]), name='b3')

            h2 = tf.nn.relu(tf.matmul(h1, W3) + b3)
            h2 = tf.nn.dropout(h2, rate=1-keep_prob, seed=self.seed6)

            # END NEW

            W2 = tf.compat.v1.get_variable('W2', [self.classifier_num_hidden_units_2, self.num_labels],
                                 initializer=tf.keras.initializers.GlorotUniform(seed=self.seed3))
            b2 = tf.Variable(tf.zeros(shape=[self.num_labels]), name='b2')

            pred_logit = tf.matmul(h2, W2) + b2
            pred_label = tf.nn.softmax(pred_logit)

        return pred_label, pred_logit

    def _adversary_model(self, pred_logits, true_labels, keep_prob):
        """Compute the adversary predictions for the protected attribute.
        """

        with tf.compat.v1.variable_scope("adversary_model"):
            if self.fairness_def == 'parity':
                W2 = tf.compat.v1.get_variable('W2', [self.num_labels, self.adversary_num_hidden_units],
                                     initializer=tf.keras.initializers.GlorotUniform(seed=self.seed4))
            elif self.fairness_def == 'equal_odds':
                W2 = tf.compat.v1.get_variable('W2', [self.num_labels*2, self.adversary_num_hidden_units],
                                     initializer=tf.keras.initializers.GlorotUniform(seed=self.seed4))

            b2 = tf.Variable(tf.zeros(shape=[self.adversary_num_hidden_units]), name='b2')

            if self.fairness_def == 'parity':
                h1 = tf.nn.relu(tf.matmul(pred_logits, W2) + b2)
            elif self.fairness_def == 'equal_odds':
                h1 = tf.nn.relu(tf.matmul(tf.concat([pred_logits, true_labels], axis=1), W2) + b2)
            h1 = tf.nn.dropout(h1, rate=1-keep_prob, seed=self.seed7)

            W3 = tf.compat.v1.get_variable('W3', [self.adversary_num_hidden_units, 1],
                                 initializer=tf.keras.initializers.GlorotUniform(seed=self.seed8))
            b3 = tf.Variable(tf.zeros(shape=[1]), name='b3')


            pred_protected_attribute_logit = tf.matmul(h1, W3) + b3
            pred_protected_attribute_label = tf.sigmoid(pred_protected_attribute_logit)

        return pred_protected_attribute_label, pred_protected_attribute_logit

    def fit(self, features_set, metadata_set):
        """Compute the model parameters of the fair classifier using gradient
        descent.
        """

        if self.seed is not None:
            np.random.seed(self.seed)
        ii32 = np.iinfo(np.int32)
        self.seed1, self.seed2, self.seed3, self.seed4, self.seed5, self.seed6, self.seed7, self.seed8 = np.random.randint(ii32.min, ii32.max, size=8)

        # Map the dataset labels to one-hot
        def one_hot(x):
            return np.eye(self.num_labels)[x]
        temp_labels = metadata_set.copy()
        label_names = sorted(temp_labels.label.unique())
        for label_int in range(len(label_names)):
            label_name = label_names[label_int]
            self.label_translate[label_int] = label_name
            temp_labels.loc[(temp_labels.label == label_name), 'label'] = label_int
        temp_labels = np.array([one_hot(x) for x in temp_labels.label])

        with tf.compat.v1.variable_scope(self.scope_name):
            num_train_samples, self.features_dim = np.shape(features_set)

            # Setup placeholders
            self.features_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, self.features_dim])
            self.protected_attributes_ph = tf.compat.v1.placeholder(tf.float32, shape=[None,1])
            self.true_labels_ph = tf.compat.v1.placeholder(tf.float32, shape=[None,self.num_labels])
            self.keep_prob = tf.compat.v1.placeholder(tf.float32)

            # Obtain classifier predictions and classifier loss
            self.pred_labels, pred_logits = self._classifier_model(self.features_ph, self.features_dim, self.keep_prob)
            pred_labels_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.true_labels_ph, logits=pred_logits))

            if self.debias:
                # Obtain adversary predictions and adversary loss
                pred_protected_attributes_labels, pred_protected_attributes_logits = self._adversary_model(pred_logits, self.true_labels_ph, self.keep_prob)
                pred_protected_attributes_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=self.protected_attributes_ph, logits=pred_protected_attributes_logits))

            # Setup optimizers with learning rates
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 0.001
            learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate, global_step,
                                                                 1000, 0.96, staircase=True)
            classifier_opt = tf.compat.v1.train.AdamOptimizer(learning_rate)
            if self.debias:
                adversary_opt = tf.compat.v1.train.AdamOptimizer(learning_rate)

            classifier_vars = [var for var in tf.compat.v1.trainable_variables() if 'classifier_model' in var.name]
            if self.debias:
                adversary_vars = [var for var in tf.compat.v1.trainable_variables() if 'adversary_model' in var.name]
                # Update classifier parameters
                adversary_grads = {var: grad for (grad, var) in adversary_opt.compute_gradients(pred_protected_attributes_loss,
                                                                                      var_list=classifier_vars)}
            normalize = lambda x: x / (tf.norm(x) + np.finfo(np.float32).tiny)

            classifier_grads = []
            for (grad,var) in classifier_opt.compute_gradients(pred_labels_loss, var_list=classifier_vars):
                if self.debias:
                    unit_adversary_grad = normalize(adversary_grads[var])
                    grad -= tf.reduce_sum(grad * unit_adversary_grad) * unit_adversary_grad
                    grad -= self.adversary_loss_weight * adversary_grads[var]
                classifier_grads.append((grad, var))
            classifier_minimizer = classifier_opt.apply_gradients(classifier_grads, global_step=global_step)

            if self.debias:
                # Update adversary parameters
                with tf.control_dependencies([classifier_minimizer]):
                    adversary_minimizer = adversary_opt.minimize(pred_protected_attributes_loss, var_list=adversary_vars)#, global_step=global_step)

            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.sess.run(tf.compat.v1.local_variables_initializer())

        if self.saved_model:
            if self.verbose:
                print('RETRIEVING SAVED MODEL: {}'.format(self.saved_model), file=sys.stderr)
            try:
                saver = tf.compat.v1.train.import_meta_graph(self.saved_model + '/model.meta')
                saver.restore(self.sess, tf.compat.v1.train.latest_checkpoint('./' + self.saved_model + '/'))
                return self
            except:
                import traceback
                print(sys.exc_info()[0], file=sys.stderr, flush=True)
                print(sys.exc_info()[1], file=sys.stderr, flush=True)
                print(traceback.print_tb(sys.exc_info()[2]), file=sys.stderr, flush=True)
                print('Failed: continuing', file=sys.stderr)

            # Begin training
        with tf.compat.v1.variable_scope(self.scope_name):
            for epoch in range(self.num_epochs):
                shuffled_ids = np.random.choice(num_train_samples, num_train_samples, replace=False)
                for i in range(num_train_samples//self.batch_size):
                    batch_ids = shuffled_ids[self.batch_size*i: self.batch_size*(i+1)]
                    batch_features = features_set.loc[batch_ids]
                    batch_labels = temp_labels[batch_ids]
                    batch_protected_attributes = np.reshape(list(metadata_set[self.protected_attribute_name].loc[batch_ids]), [-1,1])

                    batch_feed_dict = {self.features_ph: batch_features,
                                       self.true_labels_ph: batch_labels,
                                       self.protected_attributes_ph: batch_protected_attributes,
                                       self.keep_prob: 0.8}

                    if self.debias:
                        _, _, pred_labels_loss_value, pred_protected_attributes_loss_vale = self.sess.run([classifier_minimizer,
                                       adversary_minimizer,
                                       pred_labels_loss,
                                       pred_protected_attributes_loss], feed_dict=batch_feed_dict)
                        if i % 200 == 0 and self.verbose:
                            print("epoch %d; iter: %d; batch classifier loss: %f; batch adversarial loss: %f" % (epoch, i, pred_labels_loss_value,
                                                                                     pred_protected_attributes_loss_vale),
                                  file=sys.stderr, flush=True)
                    else:
                        _, pred_labels_loss_value = self.sess.run(
                            [classifier_minimizer,
                             pred_labels_loss], feed_dict=batch_feed_dict)
                        if i % 200 == 0 and self.verbose:
                            print("epoch %d; iter: %d; batch classifier loss: %f" % (
                                  epoch, i, pred_labels_loss_value),
                                  file=sys.stderr, flush=True)

        if self.saved_model:
            model_name = self.saved_model + '/model'
            if self.verbose:
                print('SAVING MODEL: {}'.format(model_name), file=sys.stderr)
            saver = tf.compat.v1.train.Saver()
            saver.save(self.sess, model_name)
            # print(self.__dict__, file=sys.stderr)

        return self

    def predict(self, features_set, metadata_set):
        """Obtain the predictions for the provided dataset using the fair
        classifier learned.
        """

        if self.seed is not None:
            np.random.seed(self.seed)

        def one_hot(x):
            return np.eye(self.num_labels)[x]
        temp_labels = metadata_set.copy()
        for label_int in self.label_translate:
            label_name = self.label_translate[label_int]
            temp_labels.loc[(temp_labels.label == label_name), 'label'] = label_int
        try:
            temp_labels = np.array([one_hot(x) for x in temp_labels.label])
        except IndexError:
            temp_labels = np.array([np.zeros(len(self.label_translate)) for x in temp_labels.label])

        num_test_samples, _ = np.shape(features_set)

        samples_covered = 0
        pred_labels = []
        while samples_covered < num_test_samples:
            start = samples_covered
            end = samples_covered + self.batch_size
            if end > num_test_samples:
                end = num_test_samples
            batch_ids = np.arange(start, end)
            batch_features = features_set.loc[batch_ids]
            batch_labels = temp_labels[batch_ids]
            batch_protected_attributes = np.reshape(list(metadata_set[self.protected_attribute_name].loc[batch_ids]), [-1,1])

            batch_feed_dict = {self.features_ph: batch_features,
                               self.true_labels_ph: batch_labels,
                               self.protected_attributes_ph: batch_protected_attributes,
                               self.keep_prob: 1.0}

            pred_labels += self.sess.run(self.pred_labels, feed_dict=batch_feed_dict).tolist()
            samples_covered += len(batch_features)

        pred_labels = np.array(pred_labels, dtype=np.float64)
        dataset_new = metadata_set.copy()
        for label_num in self.label_translate:
            dataset_new['pred_score_{}'.format(self.label_translate[label_num])] = pred_labels[:,label_num]
        dataset_new['pred_label'] = [self.label_translate[x] for x in (np.argmax(pred_labels, axis=1)).astype(np.int32).tolist()]

        return dataset_new
