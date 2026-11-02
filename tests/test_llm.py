import base64
from io import BytesIO
import math
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, Mock

import httpx
import pandas as pd
from PIL import Image
import pytest

import kairo
import kairo.llm as llm


def openai_response(text="hello", tokens=17):
    return NS(choices=[NS(message=NS(content=text))], usage=NS(completion_tokens=tokens))


@pytest.mark.parametrize("provider", ["openai", "google", "custom"])
def test_openai_compatible_routing_calls_and_streaming(monkeypatch, provider):
    client = Mock()
    constructor = Mock(return_value=client)
    monkeypatch.setattr(llm, "OpenAI", constructor)
    connector = kairo.LLMConnector(provider, "test-model", api_key="test-key", base_url="http://example.invalid/v1")
    expected_url = {"openai": None, "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
                    "custom": "http://example.invalid/v1"}[provider]
    constructor.assert_called_once_with(api_key="test-key", base_url=expected_url)
    response = openai_response()
    client.chat.completions.create.return_value = response
    assert connector.call("hello", "system", max_tokens=42) is response
    request = client.chat.completions.create.call_args.kwargs
    assert request == {"model": "test-model", "messages": kairo.create_messages(provider, "system", ["hello"], []), "max_tokens": 42}
    messages = [{"role": "user", "content": "unchanged"}]
    connector.call(messages=messages)
    assert client.chat.completions.create.call_args.kwargs["messages"] is messages
    usage = NS(completion_tokens=11)
    client.chat.completions.create.return_value = [NS(choices=[], usage=None),
        NS(choices=[NS(delta=NS(content="hel"))], usage=None), NS(choices=[NS(delta=NS(content=None))], usage=None),
        NS(choices=[NS(delta=NS(content="lo"))], usage=None), NS(choices=[], usage=usage)]
    assert list(connector.stream("question", system_prompt="instructions", max_tokens=7)) == ["hel", "lo"]
    assert kairo.get_response_text(connector.response) == "hello"
    assert kairo.count_output_tokens(provider, connector.response) == 11
    assert client.chat.completions.create.call_args.kwargs["stream_options"] == {"include_usage": True}


def test_legacy_custom_stream_retry_and_missing_usage(monkeypatch):
    client = Mock()
    monkeypatch.setattr(llm, "OpenAI", Mock(return_value=client))
    bad_response = httpx.Response(400, request=httpx.Request("POST", "http://example.invalid/v1"))
    error = llm.BadRequestError("stream_options unsupported", response=bad_response, body=None)
    client.chat.completions.create.side_effect = [error, [NS(choices=[NS(delta=NS(content="eight123"))], usage=None)]]
    connector = kairo.LLMConnector("custom", "test")
    assert list(connector.stream(messages=[{"role": "user", "content": "hello"}])) == ["eight123"]
    calls = client.chat.completions.create.call_args_list
    assert "stream_options" in calls[0].kwargs and "stream_options" not in calls[1].kwargs
    assert connector.response.usage is None
    assert kairo.count_output_tokens("custom", connector.response) == 2


@pytest.mark.parametrize("with_system", [True, False])
def test_anthropic_calls_and_streaming(monkeypatch, with_system):
    response = NS(content=[NS(type="thinking", thinking="hidden"), NS(type="text", text="hello"), NS(type="text", text=" world")],
                  usage=NS(output_tokens=25))
    stream = NS(text_stream=iter(["hello", " world"]), get_final_message=Mock(return_value=response))
    context = MagicMock()
    context.__enter__.return_value = stream
    client = Mock()
    client.messages.stream.return_value = context
    constructor = Mock(return_value=client)
    monkeypatch.setattr(llm, "Anthropic", constructor)
    connector = kairo.LLMConnector("anthropic", "test", "test-key")
    constructor.assert_called_once_with(api_key="test-key")
    system = "system instructions" if with_system else None
    assert connector.call("question", system, max_tokens=200) is response
    request = client.messages.stream.call_args.kwargs
    assert all(message["role"] != "system" for message in request["messages"])
    assert request["model"] == "test" and request["max_tokens"] == 200
    assert request.get("system") == system
    assert list(connector.stream("question", system)) == ["hello", " world"]
    assert connector.response is response
    assert kairo.get_response_text(response) == "hello world"
    assert kairo.count_output_tokens("anthropic", response) == 25


def test_invalid_provider_rejected():
    with pytest.raises(ValueError, match="Unknown provider"):
        kairo.LLMConnector("invalid", "test")


@pytest.mark.parametrize("provider", ["openai", "custom", "google", "anthropic"])
@pytest.mark.parametrize("size", [(100, 100), (1024, 768), (4096, 2048)])
def test_multimodal_message_encoding_and_documented_token_estimates(provider, size):
    buffer = BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="PNG")
    png = buffer.getvalue()
    figure = Mock()
    figure.to_image.return_value = png
    messages = kairo.create_messages(provider, "abcd", ["abcd", "efgh"], [figure])
    figure.to_image.assert_called_once_with(format="png")
    assert messages[0] == {"role": "system", "content": "abcd"}
    assert messages[1]["content"][0] == {"type": "text", "text": "abcd\n\nefgh"}
    image = messages[1]["content"][1]
    if provider == "anthropic":
        assert image["source"]["media_type"] == "image/png"
        encoded = image["source"]["data"]
        expected = min(size[0] * size[1] // 750, 1600)
    else:
        encoded = image["image_url"]["url"].split(",", 1)[1]
        if provider == "google":
            expected = 258 * math.ceil(size[0] / 768) * math.ceil(size[1] / 768)
        else:
            # Precomputed tile counts for the three test dimensions.
            expected = {(100, 100): 255, (1024, 768): 765, (4096, 2048): 1105}[size]
    assert base64.b64decode(encoded) == png
    assert kairo.count_input_tokens(provider, messages) == expected + 3
    history = kairo.get_chat_history(messages + [{"role": "assistant", "content": "answer"}])
    assert "(4 characters)" in history and "[1 plot]" in history and "answer" in history
    assert encoded not in history
    two = kairo.create_messages(provider, None, ["hello"], [figure, figure])
    assert "[2 plots]" in kairo.get_chat_history(two)
    assert kairo.count_input_tokens(provider, [{"role": "user", "content": "eight123"}]) == 2


def test_abstractions_preserve_measured_quantities(event_log, numeric_features):
    stats = kairo.compute_log_stats(event_log)
    text = kairo.abstract_log_stats(stats)
    assert "8 events of 3 cases, 3 activities and 3 resources" in text
    assert "A (4)" in text and "r1 (3)" in text
    assert "Most active resources" not in kairo.abstract_log_stats(kairo.compute_log_stats(event_log.drop(columns="org:resource")))
    for perspective in ["intra", "resource", "inter"]:
        kwargs = {"sliding_window_size": 2} if perspective == "intra" else {}
        features = getattr(kairo, f"compute_features_{perspective}")(event_log, **kwargs)
        description = kairo.abstract_features(features)
        assert {"intra": "Intra-case", "resource": "Resource", "inter": "Inter-case"}[perspective] in description
        assert f"{len(features):,} rows" in description and "Feature groups:" in description
    pca = kairo.compute_pca(numeric_features, 3)
    assert "keeps the first 2" in kairo.abstract_pca(pca, 2)
    frequencies = pd.Series([3, 1], index=["0", "1"], name="events")
    distances = pd.DataFrame([[0, 2, 5], [2, 0, 3], [5, 3, 0]], index=["0", "1", "2"], columns=["0", "1", "2"])
    colors = {0: "#ff0000", 1: "#0000ff", 2: "#888888", -1: "#eeeeee"}
    description = kairo.abstract_states(frequencies, distances, colors, [
        {"type": "silhouette", "value": .5}, {"type": "undefined", "value": float("nan")}, {"type": "text", "value": "n/a"}])
    assert "3 events (75.0%)" in description and "States without events: 2" in description
    assert "0: nearest 1 (2.00), farthest 2 (5.00)" in description
    assert "silhouette: 0.500" in description and "undefined: undefined" in description and "text: n/a" in description


def test_trajectory_and_signal_abstractions(event_log):
    states = event_log[["case:concept:name", "time:timestamp"]].assign(cluster=[0, 0, 1, 1, 1, 0, 1, 0])
    single = kairo.get_kmeans_case_trajectory(states, "a")
    assert "Trajectory of case a" in kairo.abstract_case_trajectory(single, "a")
    assert "Trajectories of 3 cases" in kairo.abstract_case_trajectories(kairo.get_kmeans_case_trajectories(states))
    windows = pd.DataFrame({"time:timestamp": pd.date_range("2025-01-01", periods=6, freq="1D"), "cluster": [0, 0, 1, 1, 0, 1]})
    description = kairo.abstract_log_trajectory(kairo.get_kmeans_log_trajectory(windows))
    assert "6 feature windows in 4 visits" in description
    assert "0: 3 feature windows (50.0%)" in description
    long = pd.DataFrame({"time:timestamp": pd.date_range("2025-01-01", periods=210, freq="1D"), "cluster": [i % 2 for i in range(210)]})
    assert "200 longest of the 210 visits" in kairo.abstract_log_trajectory(kairo.get_kmeans_log_trajectory(long))
    distributions = kairo.compute_state_distributions(windows, "1D")
    assert "6 signal windows" in kairo.abstract_state_distributions(distributions)
    for reference in ["previous", "recent", "baseline"]:
        signal = kairo.compute_drift_signal(distributions, "tv", reference, lookback=2)
        text = kairo.abstract_drift_signal(signal, "tv", reference, lookback=2)
        assert "Total variation" in text and "6 signal windows" in text
        assert f"mean {signal.drift_score.mean():.3f}" in text
