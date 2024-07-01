# pylint: disable=missing-function-docstring, missing-class-docstring, redefined-outer-name, no-self-use, unused-argument, abstract-class-instantiated, invalid-name, protected-access
import numpy as np
import pytest
from pytest_mock import MockFixture

from symaware.base import (
    Agent,
    AsyncLoopLock,
    AwarenessVector,
    CommunicationReceiver,
    CommunicationSender,
    Component,
    Controller,
    DynamicalModel,
    Entity,
    Environment,
    KnowledgeDatabase,
    MultiAgentAwarenessVector,
    MultiAgentKnowledgeDatabase,
    PerceptionSystem,
    RiskEstimator,
    TimeSeries,
    UncertaintyEstimator,
)


class PytestKnowledgeDatabase(KnowledgeDatabase):
    integer: int
    string: str
    array: list[int]
    dictionary: dict[str, int]
    ndarray: np.ndarray


class UniqueId:
    id: int = 0

    @classmethod
    def get(cls) -> int:
        res = cls.id
        cls.id += 1
        return res


@pytest.fixture(name="PatchedEnvironment")
def fixture_PatchedEnvironment(mocker: MockFixture) -> type[Environment]:
    mocker.patch.multiple(
        Environment,
        **{method: mocker.Mock() for method in Environment.__abstractmethods__},
        __abstractmethods__=set(),
    )
    return Environment


@pytest.fixture(name="PatchedComponent")
def fixture_PatchedComponent(mocker: MockFixture) -> type[Component]:
    mocker.patch.multiple(
        Component,
        **{method: mocker.Mock() for method in Component.__abstractmethods__},
        __abstractmethods__=set(),
    )
    return Component


@pytest.fixture(name="PatchedController")
def fixture_PatchedController(mocker: MockFixture) -> type[Controller]:
    mocker.patch.multiple(
        Controller,
        **{method: mocker.Mock() for method in Controller.__abstractmethods__ if method != "_compute"},
        __abstractmethods__=set(),
        _compute=lambda self: (np.zeros(3), TimeSeries()),
    )
    return Controller


@pytest.fixture(name="PatchedCommunicationSender")
def fixture_PatchedCommunicationSender(mocker: MockFixture) -> type[CommunicationSender]:
    mocker.patch.multiple(
        CommunicationSender,
        **{method: mocker.Mock() for method in CommunicationSender.__abstractmethods__},
        __abstractmethods__=set(),
    )
    return CommunicationSender


@pytest.fixture(name="PatchedCommunicationReceiver")
def fixture_PatchedCommunicationReceiver(mocker: MockFixture) -> type[CommunicationReceiver]:
    mocker.patch.multiple(
        CommunicationReceiver,
        **{method: mocker.Mock() for method in CommunicationReceiver.__abstractmethods__},
        __abstractmethods__=set(),
    )
    return CommunicationReceiver


@pytest.fixture(name="PatchedPerceptionSystem")
def fixture_PatchedPerceptionSystem(mocker: MockFixture) -> type[PerceptionSystem]:
    mocker.patch.multiple(
        PerceptionSystem,
        **{method: mocker.Mock() for method in PerceptionSystem.__abstractmethods__},
        __abstractmethods__=set(),
    )
    return PerceptionSystem


@pytest.fixture(name="PatchedRiskEstimator")
def fixture_PatchedRiskEstimator(mocker: MockFixture) -> type[RiskEstimator]:
    mocker.patch.multiple(
        RiskEstimator,
        **{method: mocker.Mock() for method in RiskEstimator.__abstractmethods__},
        __abstractmethods__=set(),
    )
    return RiskEstimator


@pytest.fixture(name="PatchedUncertaintyEstimator")
def fixture_PatchedUncertaintyEstimator(mocker: MockFixture) -> type[UncertaintyEstimator]:
    mocker.patch.multiple(
        UncertaintyEstimator,
        **{method: mocker.Mock() for method in UncertaintyEstimator.__abstractmethods__},
        __abstractmethods__=set(),
    )
    return UncertaintyEstimator


@pytest.fixture(name="PatchedDynamicalModel")
def fixture_PatchedDynamicalModel(mocker: MockFixture) -> type[DynamicalModel]:
    mocker.patch.multiple(
        DynamicalModel,
        **{method: mocker.Mock() for method in DynamicalModel.__abstractmethods__ if method != "subinputs_dict"},
        __abstractmethods__=set(),
        subinputs_dict={},
    )
    return DynamicalModel


@pytest.fixture(name="PatchedEntity")
def fixture_PatchedEntity(mocker: MockFixture) -> type[Entity]:
    mocker.patch.multiple(
        Entity,
        **{method: mocker.Mock() for method in Entity.__abstractmethods__},
        __abstractmethods__=set(),
    )
    return Entity


@pytest.fixture(name="PatchedAsyncLoopLock")
def fixture_PatchedAsyncLoopLock(mocker: MockFixture) -> type[AsyncLoopLock]:
    mocker.patch.multiple(
        AsyncLoopLock,
        **{method: mocker.Mock() for method in AsyncLoopLock.__abstractmethods__},
        __abstractmethods__=set(),
    )
    return AsyncLoopLock


class EnvironmentRequest(pytest.FixtureRequest):
    param: "AsyncLoopLock | tuple[AsyncLoopLock | None]"


@pytest.fixture(params=[0], name="environment")
def fixture_environment(request: EnvironmentRequest, PatchedEnvironment: type[Environment]) -> Environment:
    if isinstance(request.param, AsyncLoopLock):
        return PatchedEnvironment(request.param)
    if isinstance(request.param, tuple):
        return PatchedEnvironment(request.param[0])
    return PatchedEnvironment()


class AgentRequest(pytest.FixtureRequest):
    param: "int | Entity"


@pytest.fixture(params=[0], name="agent")
def fixture_agent(request: AgentRequest, entity: Entity, PatchedDynamicalModel: type[DynamicalModel]) -> Agent:
    param = request.param
    if isinstance(param, int):
        object.__setattr__(entity, "id", param)
        object.__setattr__(entity, "model", PatchedDynamicalModel(param, np.zeros(3)))
        return Agent(param, entity)
    if isinstance(param, Entity):
        return Agent(param.id, param)
    object.__setattr__(entity, "model", PatchedDynamicalModel(entity.id, np.zeros(3)))
    return Agent(entity.id, entity)


class EntityRequest(pytest.FixtureRequest):
    param: "tuple[DynamicalModel | None]"


@pytest.fixture(params=[0], name="entity")
def fixture_entity(request: EntityRequest, PatchedEntity: type[Entity]) -> Entity:
    if not isinstance(request.param, tuple):
        return PatchedEntity(UniqueId.get())
    return PatchedEntity(request.param[0].id, request.param[0])


class AwarenessVectorRequest(pytest.FixtureRequest):
    param: "int | tuple[int] | tuple[tuple[int, np.ndarray], ...]"


@pytest.fixture(params=[0], name="awareness_vector")
def fixture_awareness_vector(request: AwarenessVectorRequest) -> MultiAgentAwarenessVector:
    IDs = (UniqueId.get(),)
    state = np.zeros(3)
    if isinstance(request.param, int):
        IDs = (request.param,)
    elif isinstance(request.param, tuple) and isinstance(request.param[0], int):
        IDs = request.param
    elif isinstance(request.param, tuple) and isinstance(request.param[0], tuple):
        return {ID: AwarenessVector(ID, state) for ID, state in request.param}

    return {ID: AwarenessVector(ID, state) for ID in IDs}


class KnowledgeDatabaseRequest(pytest.FixtureRequest):
    param: "int | tuple[int] | tuple[tuple[int, PytestKnowledgeDatabase], ...]"


@pytest.fixture(params=[0], name="knowledge_database")
def fixture_knowledge_database(request: KnowledgeDatabaseRequest) -> MultiAgentKnowledgeDatabase:
    IDs = (UniqueId.get(),)
    value = PytestKnowledgeDatabase(integer=0, string="", array=[], dictionary={}, ndarray=np.zeros(1))
    if isinstance(request.param, int):
        IDs = (request.param,)
    elif isinstance(request.param, tuple) and isinstance(request.param[0], int):
        IDs = request.param
    elif isinstance(request.param, tuple) and isinstance(request.param[0], tuple):
        return {ID: value for ID, value in request.param}

    return {ID: PytestKnowledgeDatabase(**value) for ID in IDs}
