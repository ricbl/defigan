"""Scheduler for knowing which losses to use at each iteration

Provides a way to generate periodic outputs to coordinate training

by Ricardo Bigolin Lanfredi
Last modified: 2020-07-01
Project: DeFI-GAN
GNU General Public License v3.0
"""

class Schedule:
    def __init__(self, values_returned = [], list_of_iteration=[], total_epochs=None):
        n_nones_in_list_of_iteration =  sum([1 if e is None else 0 for e in list_of_iteration])
        assert(((n_nones_in_list_of_iteration==1) != (total_epochs is None)) and (n_nones_in_list_of_iteration<2))
        if n_nones_in_list_of_iteration==1:
            rest = sum([0 if e is None else e for e in list_of_iteration])
            list_of_iteration = [total_epochs-rest if e is None else e for e in list_of_iteration]
        if list_of_iteration==[] and len(values_returned)==1:
            list_of_iteration = [1]
        assert(len(list_of_iteration)==len(values_returned))
        
        self.current_index = 0
        self.current_count_in_index = 0
        self.list_of_iteration=list_of_iteration
        self.values_returned = values_returned
        
    def next(self):
        if self.current_count_in_index>=self.list_of_iteration[self.current_index]:
            self.current_index+=1
            self.current_index = self.current_index%len(self.list_of_iteration)
            self.current_count_in_index = 0
            return self.next()
        else:
            self.current_count_in_index += 1
            return self.values_returned[self.current_index]

def get_schedule(opt):
    if opt.use_old_schedule:
        return get_old_schedule(opt)
    
    #return a schedule where critic and generator are updated in every batch
    mid_schedule = Schedule([{'step_generator':True,'step_critic':True, 'step_adversarial':True}])
    return Schedule([None, mid_schedule, None], [0, None, 0], opt.nepochs)

#emulates the schedule from the "Visual Feature Attribution Using Wasserstein GANs" paper
def get_old_schedule(opt):
    middle_generator = {'step_generator':True,'step_critic':False, 'step_adversarial':False}
    middle_critic = {'step_generator':False,'step_critic':True, 'step_adversarial':True}
    schedule_1 = Schedule([middle_critic, middle_generator], [5, 1])
    schedule_2 = Schedule([middle_critic, middle_generator], [99, 1])
    return Schedule([schedule_2, schedule_1], [opt.length_initialization_old_schedule, None], opt.nepochs)