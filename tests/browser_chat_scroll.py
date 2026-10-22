"""Real streamed responses and browser scrolling; deterministic inference only."""
import argparse
import copy
import json
import os
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import expect, sync_playwright
from lazyrig.server import handler
from lazyrig.worker import Cancelled
from support import DEFAULT, FixtureWorker, Service


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('test-results/browser-chat-scroll.json'))
    args = parser.parse_args()
    report = {'scope': 'Chromium, real HTTP/SSE, deterministic streaming worker', 'checks': []}
    release = threading.Event()
    with tempfile.TemporaryDirectory() as root:
        config = copy.deepcopy(DEFAULT)
        config['data_dir'] = root
        service = Service(config)

        def factory(*worker_args):
            worker = FixtureWorker(*worker_args)
            original = worker.run

            def run(request, timeout, cancelled):
                if request['action'] in {'image', 'image_edit'}:
                    yield from original(request, timeout, cancelled)
                    return
                text = '\n\n'.join(f'Paragraph {n}: A steady reading position while this response continues.' for n in range(45))
                n = 45
                while not release.wait(.04):
                    if cancelled.is_set():
                        raise Cancelled()
                    text += f'\n\nParagraph {n}: More streamed text with **emphasis** and `code`.'
                    n += 1
                    yield {'kind': 'text', 'text': text, 'generated_tokens': n * 15,
                           'accuracy': 'estimated-test'}
                yield {'kind': 'result', 'text': text, 'generated_tokens': n * 15,
                       'accuracy': 'estimated-test'}

            worker.run = run
            return worker

        service.scheduler.factory = factory
        http = ThreadingHTTPServer(('127.0.0.1', 0), handler(service))
        threading.Thread(target=http.serve_forever, daemon=True).start()
        failures = []
        try:
            with sync_playwright() as p:
                executable = os.environ.get('CHROMIUM_EXECUTABLE')
                # Exercise the actual native thumb as well as wheel/touch input.
                browser = p.chromium.launch(ignore_default_args=['--hide-scrollbars'],
                    **({'executable_path': executable} if executable else {}))
                page = browser.new_page(viewport={'width': 1440, 'height': 900})
                page.on('pageerror', lambda e: failures.append(str(e)))
                page.goto('http://127.0.0.1:' + str(http.server_port))
                expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('nav [data-view="chat"]').click()
                page.locator('#chat-task').select_option('image')
                page.locator('#chat-prompt').fill('Create an image to keep in this conversation.')
                page.locator('#chat-send').click()
                expect(page.locator('.message-images figure img')).to_be_visible(timeout=15000)
                expect(page.locator('#chat-stop')).not_to_be_visible(timeout=15000)
                page.evaluate("() => { window.historyImage = document.querySelector('.message-images figure img'); }")
                page.locator('#chat-task').select_option('chat')
                page.locator('#chat-prompt').fill('Produce a long streamed response.')
                page.locator('#chat-send').click()
                expect(page.locator('.message.assistant .message-body p').nth(48)).to_be_attached(timeout=15000)
                box = page.locator('#chat-messages').bounding_box()
                page.mouse.move(box['x'] + box['width'] / 2, box['y'] + box['height'] / 2)

                def position():
                    return page.locator('#chat-messages').evaluate('(e) => ({top:e.scrollTop, gap:e.scrollHeight-e.scrollTop-e.clientHeight, height:e.scrollHeight})')

                initial = position()
                page.wait_for_timeout(250)
                followed = position()
                assert followed['top'] > initial['top'] and followed['gap'] <= 2, (initial, followed)
                report['checks'].append('Untouched view follows newly streamed text')

                # A small wheel movement must disengage follow immediately,
                # even while still inside the former 90-pixel near-bottom zone.
                page.mouse.wheel(0, -35)
                page.wait_for_timeout(100)
                held = position()
                page.wait_for_timeout(500)
                after = position()
                report['small_wheel'] = {'before': initial, 'held': held, 'after': after}
                assert abs(after['top'] - held['top']) <= 2, report['small_wheel']
                assert after['height'] > held['height'] and after['gap'] > 90
                expect(page.locator('#chat-latest')).to_be_visible()
                report['checks'].append('Small upward wheel movement holds its position through continuing updates')

                page.locator('#chat-latest').click()
                page.wait_for_timeout(120)
                assert position()['gap'] <= 2
                page.wait_for_timeout(250)
                assert position()['gap'] <= 2
                report['checks'].append('Latest button resumes continuous following')

                page.locator('#chat-messages').focus()
                page.keyboard.press('ArrowUp')
                page.wait_for_timeout(180)
                held = position()
                page.wait_for_timeout(350)
                assert abs(position()['top'] - held['top']) <= 2, (held, position())
                report['checks'].append('Keyboard scrolling can pause following')

                # Scrolling back to the bottom manually resumes following.
                page.mouse.move(box['x'] + box['width'] / 2, box['y'] + box['height'] / 2)
                page.mouse.wheel(0, 100000)
                page.wait_for_timeout(180)
                assert position()['gap'] <= 2
                page.wait_for_timeout(200)
                assert position()['gap'] <= 2
                report['checks'].append('Manual return to the bottom resumes following')

                bar = page.locator('#chat-messages').evaluate('''(e) => {
                    const r=e.getBoundingClientRect();
                    return {x:r.x,y:r.y,width:r.width,client:e.clientHeight,
                        content:e.scrollHeight,gutter:e.offsetWidth-e.clientWidth,top:e.scrollTop};
                }''')
                thumb = bar['client'] * bar['client'] / bar['content']
                track_y = bar['y'] + bar['client'] - thumb / 2
                track_x = bar['x'] + bar['width'] - max(2, bar['gutter']/2)
                page.mouse.move(track_x, track_y)
                page.mouse.down()
                page.mouse.move(track_x, track_y - 150, steps=12)
                page.mouse.up()
                page.wait_for_timeout(250)
                dragged = position()
                page.wait_for_timeout(350)
                held_drag = position()
                report['scrollbar_drag'] = {'before':bar,'dragged':dragged,'after':held_drag}
                assert dragged['top'] < bar['top'] - 100, report['scrollbar_drag']
                assert abs(held_drag['top'] - dragged['top']) <= 2 and held_drag['height'] > dragged['height'], report['scrollbar_drag']
                expect(page.locator('#chat-latest')).to_be_visible()
                page.locator('#chat-latest').click()
                page.wait_for_timeout(150)
                assert position()['gap'] <= 2
                report['checks'].append('Native scrollbar drag pauses following while output grows, and Latest resumes')
                page.mouse.move(box['x'] + box['width'] / 2, box['y'] + box['height'] / 2)

                page.mouse.wheel(0, -450)
                page.wait_for_timeout(200)
                held = position()
                page.locator('#chat-prompt').fill('A draft while reading above.')
                page.wait_for_timeout(200)
                assert abs(position()['top'] - held['top']) <= 2
                release.set()
                expect(page.locator('#chat-stop')).not_to_be_visible(timeout=15000)
                assert abs(position()['top'] - held['top']) <= 2, (held, position())
                assert page.evaluate('() => window.historyImage.isConnected'), 'Completion replaced an existing image with a loading placeholder'
                report['checks'].append('Composer updates and completion preserve the reading position')
                report['checks'].append('Existing generated images remain mounted through stream completion')
                page.close()

                # Exercise the native touch-scroll path in a small viewport.
                release.clear()
                mobile = browser.new_context(viewport={'width': 390, 'height': 844}, has_touch=True, is_mobile=True)
                page = mobile.new_page()
                page.on('pageerror', lambda e: failures.append(str(e)))
                page.goto('http://127.0.0.1:' + str(http.server_port))
                expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('nav [data-view="chat"]').click()
                page.locator('#chat-prompt').fill('Continue while I scroll by touch.')
                page.locator('#chat-send').click()
                expect(page.locator('.message.assistant .message-body p').nth(48)).to_be_attached(timeout=15000)
                box = page.locator('#chat-messages').bounding_box()
                cdp = mobile.new_cdp_session(page)
                x, y = box['x'] + box['width'] / 2, box['y'] + 55
                cdp.send('Input.dispatchTouchEvent', {'type': 'touchStart', 'touchPoints': [{'x': x, 'y': y}]})
                for step in range(1, 7):
                    cdp.send('Input.dispatchTouchEvent', {'type': 'touchMove', 'touchPoints': [{'x': x, 'y': y + step * 25}]})
                    page.wait_for_timeout(70)
                cdp.send('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []})
                page.wait_for_timeout(400)
                held = position()
                page.wait_for_timeout(400)
                after = position()
                report['touch'] = {'held': held, 'after': after}
                assert abs(after['top'] - held['top']) <= 2 and after['height'] > held['height'], report['touch']
                expect(page.locator('#chat-latest')).to_be_visible()
                page.locator('#chat-latest').tap()
                page.wait_for_timeout(180)
                assert position()['gap'] <= 2
                release.set()
                expect(page.locator('#chat-stop')).not_to_be_visible(timeout=15000)
                report['checks'].append('Mobile touch scroll holds position; tapping Latest resumes following')
                mobile.close()
                assert not failures, failures
                report['browser'] = browser.version
                browser.close()
        finally:
            release.set()
            http.shutdown()
            http.server_close()
            service.close()
            report['page_errors'] = failures
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
