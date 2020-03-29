from training import *
from models import *
from evaluation import *
from torch import optim
import argparse

if __name__ == '__main__':

    """
        python chatbot.py -mn "cornell_movie_dialogs_chatbot" -cl "data/cornell movie-dialogs corpus/formatted_movie_lines.txt" -ev
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-mn", "--model_name", type=str, required=True,
                    help="name of the model")
    ap.add_argument("-cl", "--corpus_location", type=str, required=True,
                    help="Directory where the input corpus is located")
    ap.add_argument("-dn", "--display_name", type=str, default='bot',
                    help="name to be displayed in the chat")
    ap.add_argument("-at", "--attention", type=str, default="dot",
                    help="attention method (dot, general, concat")
    ap.add_argument("-ld", "--load_directory", type=str, default=None,
                    help="path from the which retrieve the model (None is training from scratch)")
    ap.add_argument("-sd", "--save_directory", type=str, default=None,
                    help="path where to save the model (None is don't want to save)")
    ap.add_argument("-hs", "--hidden_size", type=int, default=512,
                    help="model hidden layers size")
    ap.add_argument("-el", "--encoder_layers", type=int, default=2,
                    help="number of layers in the encoder")
    ap.add_argument("-dl", "--decoder_layers", type=int, default=2,
                    help="number of layers in the decoder")
    ap.add_argument("-dr", "--dropout", type=float, default=0.2,
                    help="dropout probability value (0 = no dropout)")
    ap.add_argument("-bs", "--batch_size", type=int, default=64,
                    help="batch size")
    ap.add_argument("-msl", "--max_sentence_length", type=int, default=20,
                    help="max length for a sentence to consider")
    ap.add_argument("-mwc", "--min_word_count", type=int, default=3,
                    help="remove rare words that appear in total less than min_count times")
    ap.add_argument("-gc", "--gradient_clipping", type=float, default=50.0,
                    help="gradient_clipping_value")
    ap.add_argument("-tf", "--teacher_forcing", type=float, default=1.0,
                    help="teacher_forcing_ratio (0. = never, 1. = always")
    ap.add_argument("-elr", "--encoder_learning_rate", type=float, default=0.0001,
                    help="encoder learning rate")
    ap.add_argument("-dlr", "--decoder_learning_rate", type=float, default=0.0005,
                    help="decoder learning rate")
    ap.add_argument("-it", "--number_iterations", type=int, default=4000,
                    help="number of training iterations (1 batch per iteration")
    ap.add_argument("-pi", "--print_iterations", type=int, default=1,
                    help="number of iterations between each print")
    ap.add_argument("-si", "--save_iterations", type=int, default=1000,
                    help="number of iterations between each checkpoint")
    ap.add_argument("-tr", "--train", default=False, action='store_true',
                    help="if True train the model")
    ap.add_argument("-ev", "--evaluate", default=False, action='store_true',
                    help="if True evaluate the model")
    args = vars(ap.parse_args())

    is_train = args["train"]
    is_eval = args["evaluate"]

    # Configure models
    model_name = args["model_name"]
    display_name = args["display_name"]
    attn_model = args["attention"]
    hidden_size = args["hidden_size"]
    encoder_n_layers = args["encoder_layers"]
    decoder_n_layers = args["decoder_layers"]
    dropout = args["dropout"]
    batch_size = args["batch_size"]

    # data preprocessing configuration
    max_length = args["max_sentence_length"]  # maximum sentence length to consider
    min_count = args["min_word_count"]  # remove rare words that appear in total less than min_count times

    # preprocess data
    corpus_location = args["corpus_location"]

    voc, pairs = loadPrepareData(corpus_location, max_length, min_count)
    print('Data preprocessed')

    # Setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Set directory to save checkpoint
    save_dir = args["save_directory"]
    # Set checkpoint to load from; set to None if starting from scratch
    load_directory = args["load_directory"]
    # Load model if a loadFilename is provided
    if load_directory:
        # If loading a model trained on GPU to CPU
        checkpoint = torch.load(load_directory, map_location=torch.device(device))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        print("Model loaded from: ", load_directory)

    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    if load_directory:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if load_directory:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    embedding = embedding.to(device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()
    print('Models built successfully')

    # Configure training/optimization
    gradient_clipping = args["gradient_clipping"]
    teacher_forcing_ratio = args["teacher_forcing"]
    encoder_learning_rate = args["encoder_learning_rate"]
    decoder_learning_rate = args["decoder_learning_rate"]
    n_iterations = args["number_iterations"]
    print_every = args["print_iterations"]
    save_every = args["save_iterations"]

    # Initialize optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=encoder_learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=decoder_learning_rate)
    if load_directory:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)
    print('Optimizers Initialized')

    # If you have cuda, configure cuda to call
    if device == 'cuda':
        for state in encoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
                    print(state[k])
                    input()
        for state in decoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    if is_train:
        # Load batches for each iteration
        training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                            for _ in range(n_iterations)]
        print("Loaded batches")

        # Training initialization
        start_iteration = 1
        total_loss = 0
        if load_directory:
            start_iteration = checkpoint['iteration']

        # Training loop
        print("Start training")
        for iteration in range(start_iteration, n_iterations + 1):
            training_batch = training_batches[iteration - 1]
            # Extract fields from batch
            input_variable, lengths, target_variable, mask, max_target_len = training_batch

            # Run a training iteration with batch
            loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder,
                         encoder_optimizer, decoder_optimizer, batch_size, gradient_clipping,
                         teacher_forcing_ratio, device)
            total_loss += loss

            # Print progress
            if iteration % print_every == 0:
                print("Iteration: {}; Progress: {:.4f}%; Average loss: {:.4f}".format(iteration,
                                                                                              iteration / n_iterations * 100,
                                                                                              total_loss / print_every))
                total_loss = 0

            # Save checkpoint
            if iteration % save_every == 0 and save_dir:
                directory = os.path.join(save_dir, model_name,
                                         '{}-{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size, iteration))
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save({
                    'iteration': iteration,
                    'en': encoder.state_dict(),
                    'de': decoder.state_dict(),
                    'en_opt': encoder_optimizer.state_dict(),
                    'de_opt': decoder_optimizer.state_dict(),
                    'embedding': embedding.state_dict()
                }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

    if is_eval:
        print("Beginning evaluation")

        if not is_train and load_directory is None:
            print("No model detected! You will be talking with a model that has not been trained")
        # Set dropout layers to eval mode
        encoder.eval()
        decoder.eval()
        # Initialize search module
        searcher = GreedySearchDecoder(encoder, decoder)

        # Begin chatting
        input_sentence = ''
        while 1:
            try:
                # Get input sentence
                input_sentence = input('> ')
                # Check if it is quit case
                if input_sentence == 'q' or input_sentence == 'quit':
                    break
                # Normalize sentence
                input_sentence = normalizeString(input_sentence)
                # Evaluate sentence
                output_words = evaluate(searcher, voc, input_sentence, max_length, device)
                # Remove EOS and PADs
                output_words = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
                print(display_name + ':', ' '.join(output_words))

            except KeyError:
                print("Error: Encountered unknown word.")
