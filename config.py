

class Config:
	##Agent type
    agent_type="CNN" #FCN or CNN
    hidden_units= 512
    ## Frames
    skip_frames = 2
    history_length = 3
    ## Optimzation
    lr=0.001
    batch_size = 64
    n_minibatches = 800
    ## testing
    n_test_episodes = 15
    rendering = True
   
