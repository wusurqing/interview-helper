"""Offline checks only: no provider connection and no real credentials."""
import asyncio
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import server


def fake_config():
    config = json.loads((Path(__file__).parent / 'config.example.json').read_text(encoding='utf-8'))
    config.update(
        volc_access_key='FAKE_ACCESS_KEY_TEST_ONLY',
        volc_secret_key='FAKE_SECRET_KEY_TEST_ONLY',
        rtc_app_id='FAKE_APP_ID',
        rtc_token='FAKE_RTC_TOKEN_TEST_ONLY',
        doubao_api_key='FAKE_ASR_KEY_TEST_ONLY',
    )
    return config


class ProbeContractTests(unittest.TestCase):
    def test_browser_configuration_never_contains_provider_secrets(self):
        config = fake_config()
        public = server.public_config(config)
        serialized = json.dumps(public)
        for name in ('volc_access_key', 'volc_secret_key', 'doubao_api_key'):
            self.assertNotIn(config[name], serialized)
        self.assertEqual(public['appId'], config['rtc_app_id'])
        self.assertEqual(public['rtcToken'], config['rtc_token'])
        self.assertTrue(public['configured'])

    def test_session_uses_correct_room_user_and_explicit_asr_2(self):
        config = fake_config()
        options = {'aivad': True, 'silence_ms': 800, 'twopass': True}
        request = server.build_session_create(config, options)
        self.assertEqual(request['type'], 'session.create')
        session = request['session']
        transport = session['AgentConfig']['TransmitConfig']
        self.assertEqual(transport['TargetUserId'], [config['rtc_user_id']])
        self.assertEqual(transport['UserId'], config['rtc_bot_id'])
        self.assertEqual(transport['RoomId'], config['rtc_room_id'])
        provider = session['ASRConfig']['ProviderParams']
        self.assertEqual(provider['Credential']['ApiKey'], config['doubao_api_key'])
        self.assertEqual(provider['Credential']['ApiResourceId'], 'volc.seedasr.sauc.duration')
        self.assertEqual(provider['StreamMode'], 2)
        self.assertTrue(json.loads(provider['VolcanoASRParameters'])['request']['enable_nonstream'])

    def test_default_does_not_add_a_forced_expiration_or_llm(self):
        config = fake_config()
        session = server.build_session_create(config, {})['session']
        vad = session['AudioConfig']['VADConfig']
        self.assertTrue(vad['AIVAD'])
        self.assertFalse(vad['ForceEnd'])
        self.assertNotIn('ExpireTime', vad)
        self.assertNotIn('LLMConfig', session)
        self.assertNotIn('TTSConfig', session)

    def test_invalid_silence_intervals_are_rejected(self):
        for value in (-1, 0, 499, 3000, 50000, 'not-a-number'):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                server.validate_options(fake_config(), {'silence_ms': value})

    def test_disabling_aivad_also_disables_prefetch_for_a_valid_comparison(self):
        request = server.build_session_create(fake_config(), {'aivad': False})
        vad = request['session']['AudioConfig']['VADConfig']
        self.assertFalse(vad['AIVAD'])
        self.assertFalse(vad['EnablePrefetch'])

    def test_logs_redact_secrets_even_inside_error_messages(self):
        config = fake_config()
        secrets = [config[key] for key in ('volc_access_key', 'volc_secret_key', 'doubao_api_key', 'rtc_token')]
        event = {'type': 'error', 'message': ' '.join(secrets), 'nested': {'key': secrets[1]}}
        output = json.dumps(server.redact(event, secrets))
        for secret in secrets:
            self.assertNotIn(secret, output)

    def test_final_event_deduplication_does_not_merge_different_turns(self):
        seen = set()
        first = {'type': 'user.audio.transcription.done', 'conv_id': 'turn-1', 'transcript': '第一问'}
        self.assertEqual(server.process_cloud_event(first, seen)['transcript'], '第一问')
        self.assertIsNone(server.process_cloud_event(first, seen))
        second = dict(first, conv_id='turn-2', transcript='第二问')
        self.assertEqual(server.process_cloud_event(second, seen)['transcript'], '第二问')

    def test_local_access_checks_reject_other_sites_and_remote_clients(self):
        self.assertTrue(server.local_request_allowed('localhost:18765', 'http://localhost:18765', '127.0.0.1', True))
        self.assertFalse(server.local_request_allowed('localhost:18765', 'https://example.com', '127.0.0.1', True))
        self.assertFalse(server.local_request_allowed('evil.example:18765', None, '127.0.0.1'))
        self.assertFalse(server.local_request_allowed('localhost:18765', None, '192.0.2.1'))

    def test_create_app_suppresses_websockets_logger_under_factory_mode(self):
        ws_logger = logging.getLogger('websockets')
        prev_level = ws_logger.level
        try:
            ws_logger.setLevel(logging.DEBUG)
            server.create_app()
            self.assertGreaterEqual(ws_logger.level, logging.WARNING)
        finally:
            ws_logger.setLevel(prev_level)

    def test_run_ps1_passes_warning_log_level(self):
        script = (Path(__file__).parent / 'run.ps1').read_text(encoding='utf-8')
        self.assertIn('--factory', script)
        self.assertIn('--log-level warning', script)

    def test_connect_mmpg_passes_silent_wire_logger(self):
        captured = {}

        async def fake_connect(url, **kwargs):
            captured['url'] = url
            captured['kwargs'] = kwargs
            return FakeCloud()

        from websockets.asyncio import client as ws_client

        with mock.patch.object(ws_client, 'connect', side_effect=fake_connect):
            asyncio.run(server.connect_mmpg(fake_config()))

        wire_logger = captured['kwargs'].get('logger')
        self.assertIs(wire_logger, server._WIRE_LOGGER)
        self.assertGreater(wire_logger.level, logging.CRITICAL)
        self.assertFalse(wire_logger.propagate)


class FakeCloud:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.closed = False

    async def send(self, payload):
        self.sent.append(json.loads(payload))

    async def recv(self):
        return json.dumps(await self.incoming.get())

    async def close(self):
        self.closed = True


class ProbeLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_late_manual_mark_never_attaches_to_the_next_turn(self):
        cloud = FakeCloud()
        messages = []
        created = asyncio.Event()
        first_done = asyncio.Event()
        second_done = asyncio.Event()

        async def connect(_config):
            return cloud

        async def send(message):
            messages.append(message)
            if message.get('event', {}).get('type') == 'user.conversation.created':
                created.set()
            if message['type'] == 'done':
                (first_done if message['conv_id'] == 'c1' else second_done).set()

        with tempfile.TemporaryDirectory() as folder:
            run = server.ProbeRun(fake_config(), {}, send, logs_dir=folder, connector=connect)
            task = asyncio.create_task(run.run())
            try:
                await cloud.incoming.put({'type': 'session.created'})
                await cloud.incoming.put({'type': 'user.conversation.created', 'conv_id': 'c1'})
                await asyncio.wait_for(created.wait(), 2)
                await run.mark_end()
                await cloud.incoming.put({'type': 'user.audio.transcription.done', 'conv_id': 'c1', 'transcript': '第一问'})
                await asyncio.wait_for(first_done.wait(), 2)
                await run.mark_end()  # This click is too late to label c1.
                await cloud.incoming.put({'type': 'user.conversation.created', 'conv_id': 'c2'})
                await cloud.incoming.put({'type': 'user.audio.transcription.done', 'conv_id': 'c2', 'transcript': '第二问'})
                await asyncio.wait_for(second_done.wait(), 2)
                done = [m for m in messages if m['type'] == 'done']
                self.assertIsNotNone(done[0]['since_mark_ms'])
                self.assertIsNone(done[1]['since_mark_ms'])
                marks = [m for m in messages if m['type'] == 'mark_end_ack']
                self.assertTrue(marks[0]['associated'])
                self.assertFalse(marks[1]['associated'])
            finally:
                await run.stop()
                await task

    async def test_simulated_events_are_deduplicated_and_stop_closes_session(self):
        cloud = FakeCloud()
        messages = []
        duplicate_seen = asyncio.Event()

        async def connect(_config):
            return cloud

        async def send(message):
            messages.append(message)
            if message.get('duplicate_done'):
                duplicate_seen.set()

        with tempfile.TemporaryDirectory() as folder:
            run = server.ProbeRun(fake_config(), {}, send, logs_dir=folder, connector=connect)
            task = asyncio.create_task(run.run())
            await cloud.incoming.put({'type': 'session.created'})
            await cloud.incoming.put({'type': 'user.audio.transcription.segment', 'conv_id': 'c1', 'transcript': '如果缓存失效'})
            final = {'type': 'user.audio.transcription.done', 'conv_id': 'c1', 'transcript': '如果缓存失效，你会怎么办？'}
            await cloud.incoming.put(final)
            await cloud.incoming.put(final)
            await asyncio.wait_for(duplicate_seen.wait(), 2)
            await run.stop()
            await task
            self.assertEqual(len([m for m in messages if m['type'] == 'done']), 1)
            self.assertTrue(cloud.closed)
            self.assertEqual(len([m for m in cloud.sent if m['type'] == 'session.close']), 1)
            self.assertEqual(messages[-1]['status'], 'stopped')
            logs = list(Path(folder).glob('*.jsonl'))
            self.assertEqual(len(logs), 1)
            data = logs[0].read_text(encoding='utf-8')
            for name in ('volc_access_key', 'volc_secret_key', 'doubao_api_key', 'rtc_token'):
                self.assertNotIn(fake_config()[name], data)

    async def test_provider_connection_error_does_not_echo_secret(self):
        messages = []

        async def connect(_config):
            raise RuntimeError('failed with credential ' + fake_config()['volc_secret_key'])

        async def send(message):
            messages.append(message)

        with tempfile.TemporaryDirectory() as folder:
            run = server.ProbeRun(fake_config(), {}, send, logs_dir=folder, connector=connect)
            await run.run()
            self.assertEqual(messages[-1]['status'], 'stopped')
            self.assertTrue(any(m.get('code') == 'cloud_connection_failed' for m in messages))
            self.assertNotIn(fake_config()['volc_secret_key'], json.dumps(messages))


if __name__ == '__main__':
    unittest.main(verbosity=2)
