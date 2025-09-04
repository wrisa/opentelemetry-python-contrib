from uuid import uuid4

import pytest

from opentelemetry import trace
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.genai.handler import (
    TelemetryHandler,
    llm_start,
    llm_stop,
)
from opentelemetry.util.genai.types import (
    ChatGeneration,
    Message,
)


@pytest.fixture(name="span_exporter")
def span_exporter_fixture():
    """Set up telemetry providers for testing"""
    # Set up in-memory span exporter to capture spans
    memory_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(memory_exporter))

    # Set the tracer provider
    trace.set_tracer_provider(tracer_provider)

    yield memory_exporter

    # Cleanup
    memory_exporter.clear()
    # Reset to default tracer provider
    trace.set_tracer_provider(trace.NoOpTracerProvider())


@pytest.mark.usefixtures("span_exporter")
def test_llm_start_and_stop_creates_span(
    request: pytest.FixtureRequest,
):
    run_id = uuid4()
    message = Message(content="hello world", type="Human", name="message name")
    chat_generation = ChatGeneration(content="hello back", type="AI")

    # Start and stop LLM invocation
    llm_start(
        [message], run_id=run_id, custom_attr="value", system="test-system"
    )
    invocation = llm_stop(
        run_id, chat_generations=[chat_generation], extra="info"
    )

    # Get the spans that were created
    exporter: InMemorySpanExporter = request.getfixturevalue("span_exporter")
    spans = exporter.get_finished_spans()

    # Verify span was created
    assert len(spans) == 1
    span = spans[0]

    # Verify span properties
    assert span.name == "test-system.chat"
    assert span.kind == trace.SpanKind.CLIENT

    # Verify span attributes
    assert span.attributes.get("gen_ai.operation.name") == "chat"
    assert span.attributes.get("gen_ai.system") == "test-system"
    # Add more attribute checks as needed

    # Verify span timing
    assert span.start_time is not None
    assert span.end_time is not None
    assert span.end_time > span.start_time

    # Verify invocation data
    assert invocation.run_id == run_id
    assert invocation.attributes.get("custom_attr") == "value"
    assert invocation.attributes.get("extra") == "info"


def test_structured_logs_emitted():
    # Configure in-memory log exporter and provider
    log_exporter = InMemoryLogExporter()
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(
        SimpleLogRecordProcessor(log_exporter)
    )

    # Build a dedicated TelemetryHandler using our logger provider
    handler = TelemetryHandler(
        emitter_type_full=True,
        logger_provider=logger_provider,
    )

    run_id = uuid4()
    message = Message(content="hello world", type="user", name="msg")
    generation = ChatGeneration(
        content="hello back",
        type="assistant",
        finish_reason="stop",
    )

    # Start and stop via the handler (emits logs at start and finish)
    handler.start_llm(
        [message], run_id=run_id, system="test-system", framework="pytest"
    )
    handler.stop_llm(run_id, chat_generations=[generation])

    # Collect logs
    logs = log_exporter.get_finished_logs()
    # Expect one input-detail log and one choice log
    assert len(logs) == 2
    records = [ld.log_record for ld in logs]

    # Assert the first record contains structured details for the input message
    # Note: order of records is exporter-specific; sort by event.name for stability
    records_by_event = {
        rec.attributes.get("event.name"): rec for rec in records
    }

    input_rec = records_by_event["gen_ai.client.inference.operation.details"]
    assert input_rec.attributes.get("gen_ai.provider.name") == "test-system"
    assert input_rec.attributes.get("gen_ai.framework") == "pytest"
    assert input_rec.body == {
        "type": "user",
        "content": "hello world",
    }

    choice_rec = records_by_event["gen_ai.choice"]
    assert choice_rec.attributes.get("gen_ai.provider.name") == "test-system"
    assert choice_rec.attributes.get("gen_ai.framework") == "pytest"
    assert choice_rec.body == {
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "type": "assistant",
            "content": "hello back",
        },
    }
