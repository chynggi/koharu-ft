//! X11 input handling that tauri-runtime-cef's GTK4 toplevel does not provide.
//!
//! The runtime hosts CEF as foreign X11 children of a GTK4 window, so every
//! pointer and key event goes to CEF and GDK never sees them. GDK therefore
//! keeps the keyboard focus on its own focus window, and winit's
//! `drag_window`/`drag_resize_window` fail for lack of a GDK button press
//! (koharu-rs/koharu#1147). Both are done here directly against the X server,
//! the way the previous winit X11 backend did.

use anyhow::{Context as _, Result, bail};
use raw_window_handle::{HasWindowHandle as _, RawWindowHandle};
use serde::Deserialize;
use tauri::{AppHandle, Manager as _};
use tauri_runtime_cef::CefRuntime;
use x11_dl::xlib;

/// Edge or corner a window resize starts from.
#[derive(Debug, Clone, Copy, Deserialize)]
pub enum ResizeDirection {
    North,
    NorthEast,
    East,
    SouthEast,
    South,
    SouthWest,
    West,
    NorthWest,
}

/// Moves the X input focus onto the browser's native window.
///
/// `BrowserHost::set_focus` only updates Chromium's focus state here and
/// leaves the X focus with GDK, hence the direct request.
pub(crate) fn focus_cef_window(browser: &cef::Browser) {
    use cef::{ImplBrowser as _, ImplBrowserHost as _};

    let Some(host) = browser.host() else {
        return;
    };
    let window = host.window_handle();
    let _ = with_display(|xlib, display| unsafe {
        // X only accepts focus on a viewable window.
        let mut attributes: xlib::XWindowAttributes = std::mem::zeroed();
        if (xlib.XGetWindowAttributes)(display, window, &mut attributes) != 0
            && attributes.map_state == xlib::IsViewable
        {
            (xlib.XSetInputFocus)(display, window, xlib::RevertToParent, xlib::CurrentTime);
        }
    });
}

/// Hands an interactive move (`None`) or resize of the main window to the
/// window manager through EWMH `_NET_WM_MOVERESIZE`, for the pointer button
/// that is currently held.
pub fn begin_move_resize(
    app: &AppHandle<CefRuntime>,
    direction: Option<ResizeDirection>,
) -> Result<()> {
    let window = app
        .get_webview_window("main")
        .context("the main window is unavailable")?;
    let window = match window.window_handle()?.as_raw() {
        RawWindowHandle::Xlib(handle) => handle.window,
        RawWindowHandle::Xcb(handle) => handle.window.get().into(),
        other => bail!("expected an X11 window handle, got {other:?}"),
    };
    // _NET_WM_MOVERESIZE_SIZE_* and _NET_WM_MOVERESIZE_MOVE.
    let action: std::ffi::c_long = match direction {
        Some(ResizeDirection::NorthWest) => 0,
        Some(ResizeDirection::North) => 1,
        Some(ResizeDirection::NorthEast) => 2,
        Some(ResizeDirection::East) => 3,
        Some(ResizeDirection::SouthEast) => 4,
        Some(ResizeDirection::South) => 5,
        Some(ResizeDirection::SouthWest) => 6,
        Some(ResizeDirection::West) => 7,
        None => 8,
    };

    with_display(|xlib, display| unsafe {
        let root = (xlib.XDefaultRootWindow)(display);
        let (mut root_return, mut child) = (0, 0);
        let (mut root_x, mut root_y, mut x, mut y) = (0, 0, 0, 0);
        let mut mask = 0;
        (xlib.XQueryPointer)(
            display,
            root,
            &mut root_return,
            &mut child,
            &mut root_x,
            &mut root_y,
            &mut x,
            &mut y,
            &mut mask,
        );
        if mask & xlib::Button1Mask == 0 {
            // The button was released before the request arrived; starting a
            // move now would leave the window stuck to the pointer.
            return;
        }

        // The window manager has to grab the pointer, which the implicit grab
        // from the press on the CEF window would prevent.
        (xlib.XUngrabPointer)(display, xlib::CurrentTime);

        let message_type = (xlib.XInternAtom)(display, c"_NET_WM_MOVERESIZE".as_ptr(), xlib::False);
        let mut event: xlib::XEvent = std::mem::zeroed();
        event.client_message = xlib::XClientMessageEvent {
            type_: xlib::ClientMessage,
            serial: 0,
            send_event: xlib::True,
            display,
            window,
            message_type,
            format: 32,
            data: xlib::ClientMessageData::from([
                root_x as std::ffi::c_long,
                root_y as std::ffi::c_long,
                action,
                xlib::Button1 as std::ffi::c_long,
                // Source indication: a normal application.
                1,
            ]),
        };
        (xlib.XSendEvent)(
            display,
            root,
            xlib::False,
            xlib::SubstructureRedirectMask | xlib::SubstructureNotifyMask,
            &mut event,
        );
    })
}

/// Runs `f` on a private Xlib connection that is closed afterwards.
fn with_display(f: impl FnOnce(&xlib::Xlib, *mut xlib::Display)) -> Result<()> {
    let xlib = xlib::Xlib::open().context("failed to load Xlib")?;
    // SAFETY: the display is only used by `f` and closed right after it.
    unsafe {
        let display = (xlib.XOpenDisplay)(std::ptr::null());
        if display.is_null() {
            bail!("failed to open the X display");
        }
        f(&xlib, display);
        (xlib.XCloseDisplay)(display);
    }
    Ok(())
}
