@contextlib.contextmanager
    def freeze_critic_but_allow_backward(self):
        critic = self.critic  # <= 내부에서 참조
        # --- BN만 eval로, 파라미터 freeze ---
        bn_layers = []
        bn_flags  = []
        for m in critic.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                bn_layers.append(m)
                bn_flags.append(m.training)
                m.eval()

        req_flags = [p.requires_grad for p in critic.parameters()]
        for p in critic.parameters():
            p.requires_grad_(False)

        was_training = critic.training
        critic.train(True)  # cuDNN RNN backward 허용
        try:
            yield
        finally:
            for p, f in zip(critic.parameters(), req_flags):
                p.requires_grad_(f)
            for m, f in zip(bn_layers, bn_flags):
                m.train(f)
            critic.train(was_training)

    def _actor_step(self, s):
        self.actor.train()
        with self.freeze_critic_but_allow_backward():  # <= 인자 없이 호출
            a01  = self.actor(s)
            q    = self.critic(s, a01)     # 그래프 유지
            loss = -q.mean()

            self.opt_actor.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            self.opt_actor.step()
        return loss.item()


        