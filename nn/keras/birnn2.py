__author__ = 'quynhdo'
from keras.layers import Recurrent
import theano.tensor as T

class Transparent(Recurrent):
    '''
        This is a full-pass layer, means the output is just the same with the input
        This layer is used for layer connecting purpose
    '''
    def __init__(self, input_dim):
        super(Transparent, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.return_sequences = True
        self.input = T.tensor3()
    def get_output(self, train=False):
        return self.get_input(train)

class Bidirectional(Recurrent):
    """
    Construct a bidirectional RNN out of an underlying RNN class. Traverses
    the input sequence both forwards and backwards and then concatenates RNN
    outputs from both directions.
    """

    def __init__(self, rnn_class, input_dim, output_dim=128, return_sequences=False, **kwargs):
        """
        All extra arguments are passed to the `rnn_class` constructor.
        """
        super(Bidirectional, self).__init__()
        self.rnn_class = rnn_class
        self.input_dim = input_dim
        if output_dim % 2 != 0:
            # since we're splitting the output dim between two underlying
            # RNNs, the total must be divisible by 2
            raise ValueError(
                "Output dimension of bidirectional RNN must be even")
        self.output_dim = output_dim
        self.return_sequences = return_sequences
        self.kwargs = kwargs

    #    self.inputlayer = Transparent(input_dim=self.input_dim)
    #    self.input = self.inputlayer.input
        self.forward_model = rnn_class(
            input_dim=self.input_dim,
            output_dim=self.output_dim // 2,
            return_sequences=self.return_sequences,
            go_backwards=False,
            **kwargs)
        self.backward_model = rnn_class(
            input_dim=self.input_dim,
            output_dim=self.output_dim // 2,
            return_sequences=self.return_sequences,
            go_backwards=True,
            **kwargs)

        self.forward_model.set_previous(self.inputlayer)
        self.backward_model.set_previous(self.inputlayer)

        self.layers = [self.inputlayer, self.forward_model, self.backward_model]
        self.params = []
        self.regularizers = []
        self.constraints = []
        for l in self.layers:
            params, regs, consts = l.get_params()
            self.regularizers += regs
            # params and constraints have the same size
            for p, c in zip(params, consts):
                if p not in self.params:
                    self.params.append(p)
                    self.constraints.append(c)

    def get_params(self):
        return self.params, self.regularizers, self.constraints

    def set_previous(self, layer):
        self.inputlayer.set_previous(layer)

    def get_output(self, train=False):
        forward_output = self.forward_model.get_output(train)
        backward_output = self.backward_model.get_output(train)
        if self.return_sequences:
            # reverse the output of the backward model along the
            # time dimension so that it's aligned with the forward model's
            # output
            backward_output = backward_output[:, ::-1, :]
        # both forward_output and backward_output have shapes like
        # (n_samples, n_timesteps, output_dim) in the case of self.return_sequences=True
        # or otherwise like (n_samples, output_dim)
        # In either case, concatenate the two outputs to get a final dimension
        # of output_dim * 2
        return T.concatenate(
            [forward_output, backward_output],
            axis=-1)

    def get_config(self):
        config_dict = {
            "name": self.__class__.__name__,
            "rnn_class": self.rnn_class.__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }
        config_dict.update(self.kwargs)
        return config_dict