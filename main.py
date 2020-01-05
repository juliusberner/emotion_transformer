
from emotion_transformer.core import EmotionModel, get_args, main

if __name__ == '__main__':
    hparams = get_args(EmotionModel)
    hparams = hparams.parse_args()

    if hparams.mode == 'default':
        main(hparams)
    elif hparams.mode == 'hparams_search':
        if hparams.gpus == 0:
            hparams.optimize_parallel_cpu(main, nb_trials=20, nb_workers=1)
        else:
            hparams.optimize_parallel_gpu(main, nb_trials=20, gpus = list(range(hparams.gpus)))
