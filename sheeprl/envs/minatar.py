import gymnasium as gym


class MinAtarWrapper(gym.Wrapper):
    def __init__(self, id, render_mode, use_minimal_action_set, *args, **kwargs):
        self.env = gym.make(
            f"MinAtar/{id}",
            render_mode=render_mode,
            use_minimal_action_set=use_minimal_action_set,
        )
        super().__init__(self.env)
        FPS = 30
        self._metadata = {"render_fps": FPS}
        self.metadata = {"render_fps": FPS}
        self.frames_per_sec = FPS
        self.render_fps = FPS

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        render = self.render()
        return render, {}

    def step(self, action):
        return self.env.step(action)

    def render(self):
        render = self.env.render()
        # convert redner from float to uint8
        if self.env.render_mode == "rgb_array":
            render = (render * 255).astype("uint8")
        return render

    def close(self) -> None:
        return self.env.close()
