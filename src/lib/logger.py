import wandb


class WandbLogger:
    def __init__(self, key, project):
        wandb.login(key=key)
        wandb.init(project=project)

    def log_metrics(self, info):
        wandb.log(info)

    def log_spec_and_audio(self, transcript, pred_spec, true_spec,
                           pred_audio, true_audio):
        wandb.log(
            {
                'Pred Audio': wandb.Audio(pred_audio.squeeze().numpy(), sample_rate=22050, caption=transcript),
                'True Audio': wandb.Audio(true_audio.squeeze().numpy(), sample_rate=22050, caption=transcript),
                'Pred Spec': wandb.Image(pred_spec.squeeze(), caption=transcript),
                'True Spec': wandb.Image(true_spec.squeeze(), caption=transcript)
            }
        )
