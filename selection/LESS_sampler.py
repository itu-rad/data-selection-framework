from selection.selectivesampler import SelectiveSampler


class LESSBasedSampler(SelectiveSampler):
    """Sampler which utilizes LESS data selection method.
       This sampler can only be utilized with LORA """

    """ LESS utilizes 5% of data samples for warmup training. This can be change for experimentation"""
    def pre_epoch(self, percentage=0.05) -> None:
        """Set mask to select all samples before each epoch starts"""
        # 
        
        
        
        # n = len(self.dataset)
        # mask = 
        # self.set_mask(mask)

    def post_epoch(self) -> None:
        """No-op post epoch hook"""
        pass

    def on_batch(self, idx: int, batch: dict) -> None:
        """No-op batch hook"""
        pass

    def after_forward(self, idx: int, batch: dict, current_loss: float) -> None:
        """No-op after forward hook"""
        pass