from plato.controller.basic_controller import BasicController
from plato.agent.component.user_simulator.agenda_based_user_simulator.\
    agenda_based_us import AgendaBasedUS
from plato.agent.conversational_agent.conversational_generic_agent import \
    ConversationalGenericAgent
from plato.agent.conversational_agent.conversational_single_agent import \
    ConversationalSingleAgent
from plato.controller import controller
from slm_lab.agent import Agent, Body

def encode_state(state,requestable_slots):
    """
    Encodes the dialogue state into a vector.

    :param state: the state to encode
    :return: int - a unique state encoding
    """

    temp = [int(state.is_terminal_state), int(state.system_made_offer)]
    for value in state.slots_filled.values():
        # This contains the requested slot
        temp.append(1) if value else temp.append(0)

    for r in requestable_slots:
        temp.append(1) if r == state.requested_slot else temp.append(0)

    return temp


if __name__ == "__main__":
    ctrl = BasicController()
    config = '/Users/andy/Ftech/plato-research-dialogue-system/plato/example/config/application/CamRest_model_reinforce_policy_train.yaml'
    configuration = ctrl.arg_parse(['_', '-config', config])['cfg_parser']
    ca = ConversationalSingleAgent(configuration)
    global_args = configuration['GENERAL']['global_arguments']
    user_simulator_args = configuration['AGENT_0']['USER_SIMULATOR']['arguments']
    # if 'USER_SIMULATOR' in configuration['AGENT_0']:
    #     # Agent 0 simulator configuration
    #     if 'package' in \
    #         configuration['AGENT_0']['USER_SIMULATOR'] and \
    #             'class' in \
    #             configuration['AGENT_0']['USER_SIMULATOR']:
    #         if 'arguments' in \
    #                 configuration['AGENT_0']['USER_SIMULATOR']:
    #             user_simulator_args =\
    #                 configuration[
    #                     'AGENT_0']['USER_SIMULATOR']['arguments']

    #         user_simulator_args.update(global_args)
            
    #         user_simulator = \
    #             ConversationalGenericAgent.load_module(
    #                 configuration['AGENT_0']['USER_SIMULATOR'][
    #                     'package'],
    #                 configuration['AGENT_0']['USER_SIMULATOR'][
    #                     'class'],
    #                 user_simulator_args
    #             )

    #         if hasattr(user_simulator, 'nlu'):
    #             USER_SIMULATOR_NLU = user_simulator.nlu
    #         if hasattr(user_simulator, 'nlg'):
    #             USER_SIMULATOR_NLG = user_simulator.nlg
    #     else:
    #         # Fallback to agenda based simulator with default settings
    #         user_simulator = AgendaBasedUS(
    #             user_simulator_args
    #         )
    #start the dialogue 
    #initialize the user simulator
    # print(ca.dialogue_manager.policy.encode_state(obs))
    # print(ca.dialogue_manager.policy.encode_action(action)) 
    ca.initialize()
    state = ca.start_dialogue()
    while not ca.terminated():
       action, state, next_state, reward, done =  ca.continue_dialogue()

    ca.end_dialogue()
