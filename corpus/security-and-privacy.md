# Security and Privacy

Nimbus Notes is built so that your notes stay yours. This page explains where your
data lives, how it is protected, and the controls you have.

## Where your data is stored

All Nimbus Notes customer data is stored in data centers located in Frankfurt,
Germany. Data is encrypted in transit (TLS 1.3) and at rest (AES-256). We do not
store your notes in any other region; if you require data residency elsewhere,
that is not currently supported.

## Two-factor authentication

You can — and should — enable two-factor authentication (2FA) under Settings ->
Security. Nimbus Notes supports authenticator apps (TOTP) and hardware security
keys (WebAuthn). SMS codes are intentionally not supported, because SIM-swapping
makes them weak. When you enable 2FA, you are shown ten one-time recovery codes;
store them somewhere safe, because they are the only way back into your account
if you lose your device.

## Deleting notes and the Trash

Deleting a note does not destroy it immediately. Deleted notes are moved to Trash
and kept there for 30 days, during which you can restore them with one click from
the Trash view. After 30 days they are permanently and irreversibly removed. You
can also empty the Trash manually at any time to delete sooner.

## Sharing and permissions

Any note or notebook can be shared with a link. Share links can be set to
view-only or edit, and can optionally require a password or expire after a set
number of days. Team plan admins can disable public link sharing organization-wide
from the admin console.

## Account data export and deletion

You can export all of your data at once from Settings -> Privacy -> Export My
Data, which produces a single downloadable archive. Requesting account deletion
from the same page erases your notes within 30 days, except where a longer
retention is legally required.
