
from emotion_transformer.lightning import EmotionModel, get_args, main

if __name__ == '__main__':
    hparams = get_args(EmotionModel)
    hparams = hparams.parse_args()

    if hparams.mode == 'default':
        main(hparams)
    elif hparams.mode == 'hparams_search':
        if hparams.gpus:
            hparams.optimize_parallel_gpu(main, max_nb_trials=20, 
                                          gpu_ids = [gpus for gpus in hparams.gpus.split(' ')])
        else:
            hparams.optimize_parallel_cpu(main, nb_trials=20, nb_workers=4)
