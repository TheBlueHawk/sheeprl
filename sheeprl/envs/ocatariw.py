import gymnasium as gym
import numpy as np
from ocatari.core import OCAtari
from gymnasium import spaces


class OCAtariWrapper(gym.Wrapper):
    """OCAtari Environment that behaves like a gymnasium environment and passes env_check. Based on RAM, the observation space is object-centric.
    More specifically it is a list position history informations of objects detected. No hud information.
    The observation space is a vector where every object position history has a fixed place.
    If an object is not detected its information entries are set to 0.
    """

    def __init__(self, id: str) -> None:
        self.ocatari_env = OCAtari(env_name=id, mode="vision", hud=True, render_mode="rgb_array")
        super().__init__(self.ocatari_env)
        self.reference_list = self._init_ref_vector()
        self.current_vector = np.zeros(4 * len(self.reference_list), dtype=np.float32)


    @property
    def observation_space(self):
        # fix to include pixel observations
        vl = len(self.reference_list) * 4
        return spaces.Dict({
            "rgb": spaces.Box(0, 255, (3, 64, 64), np.uint8),
            "objects_position": spaces.Box(low=-(2**63), high=2**63 - 2, shape=(vl,), dtype=np.float32)
        })

    @property
    def action_space(self):
        return self.ocatari_env.action_space

    def step(self, *args, **kwargs):
        #print("step called")
        obs, reward, truncated, terminated, info = self.ocatari_env.step(*args, **kwargs)
        #print("step done")
        converted_obs = self._convert_obs(obs)
        #print("converted obs")
        return converted_obs, reward, truncated, terminated, info

    def reset(self, *args, **kwargs):
        #print("reset called")
        obs, info = self.ocatari_env.reset(*args, **kwargs)
        #print("reset done")
        converted_obs = self._convert_obs(obs)
        #print("converted obs")
        return converted_obs, info
    
    def _convert_obs(self, rgb_obs):
        self._obj2vec()
        return {
            "rgb": rgb_obs,
            "objects_position": self.current_vector
        }

    def render(self, *args, **kwargs):
        #print("render called")
        return self.ocatari_env.render(*args, **kwargs)

    def close(self, *args, **kwargs):
        #print("close called")
        return self.ocatari_env.close(*args, **kwargs)

    def _init_ref_vector(self):
        #print("init ref vector")
        reference_list = []
        obj_counter = {}       
        for o in self.ocatari_env.max_objects:
            if o.category not in obj_counter.keys():
                obj_counter[o.category] = 0
            obj_counter[o.category] += 1
        for k in list(obj_counter.keys()):
            reference_list.extend([k for i in range(obj_counter[k])])
        #print("init ref vector done", reference_list)
        return reference_list

    def _obj2vec(self):
        #print("obj2vec called")
        temp_ref_list = self.reference_list.copy()
        #print("self.ocatari_env.objects", self.ocatari_env.objects)
        for o in self.ocatari_env.objects:  # populate out_vector with object instance
            if o.category not in temp_ref_list:
                continue
            idx = temp_ref_list.index(o.category)  # at position of first category occurance
            start = idx * 4
            flat = [item for sublist in o.h_coords for item in sublist]
            self.current_vector[start : start + 4] = flat  # write the slice
            temp_ref_list[idx] = ""  # remove reference from reference list
        for i, d in enumerate(temp_ref_list):
            if d != "":  # fill not populated category instances wiht 0.0's
                self.current_vector[i * 4 : i * 4 + 4] = [0.0, 0.0, 0.0, 0.0]
        #print("obj2vec done")
