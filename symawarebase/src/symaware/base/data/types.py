from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    # String type hinting to support python 3.9
    import sys
    from typing import Iterable

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias
    from symaware.base import (
        Agent,
        CommunicationReceiver,
        CommunicationSender,
        Controller,
        Environment,
        MultiAgentAwarenessVector,
        MultiAgentKnowledgeDatabase,
        PerceptionSystem,
        RiskEstimator,
        UncertaintyEstimator,
    )

Identifier: "TypeAlias" = int


class SymawareConfig(TypedDict):
    agent: "tuple[Agent] | list[Agent]"
    controller: "tuple[Controller] | list[Controller]"
    knowledge_database: "tuple[MultiAgentKnowledgeDatabase] | list[MultiAgentKnowledgeDatabase]"
    awareness_vector: "tuple[MultiAgentAwarenessVector] | list[MultiAgentAwarenessVector]"
    risk_estimator: "tuple[RiskEstimator] | list[RiskEstimator]"
    uncertainty_estimator: "tuple[UncertaintyEstimator] | list[UncertaintyEstimator]"
    communication_sender: "tuple[CommunicationSender] | list[CommunicationSender]"
    communication_receiver: "tuple[CommunicationReceiver] | list[CommunicationReceiver]"
    perception_system: "tuple[PerceptionSystem] | list[PerceptionSystem]"
    environment: "Environment"
