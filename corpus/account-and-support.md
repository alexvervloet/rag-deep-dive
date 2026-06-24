# Account and Support

## Changing your email or password

Update your email address or password from Settings -> Account. Changing your
password signs out all other devices as a security precaution; you'll need to sign
back in on each one. Changing your email sends a confirmation link to the new
address, and the change only takes effect once you click it.

## Recovering a lost account

If you can't sign in, use "Forgot password" on the login screen to receive a reset
link. If you've also lost access to your two-factor device, use one of the ten
recovery codes you saved when enabling 2FA. If you have neither the password nor a
recovery code, support cannot restore access — this is a deliberate security
tradeoff, not an oversight.

## Contacting support

Email support@nimbusnotes.example for help. Support operates Monday to Friday,
9:00–17:00 Central European Time, and aims to reply within one business day. Team
plan customers get priority support with a four-hour target response during those
hours.

There is no phone support. Complex issues are handled over email so there is a
written record for both sides.

## The Nimbus Notes API

Plus and Team plans include access to a read/write HTTP API for building your own
integrations. Generate an API token from Settings -> Developer. The API is rate
limited to 120 requests per minute per token; exceeding it returns HTTP 429, and
you should retry after the number of seconds given in the `Retry-After` header.

API tokens carry the same permissions as your account, so treat them like
passwords. You can revoke a token at any time from the same page, which takes
effect within about a minute.

## Status and incidents

Live service status is published at status.nimbusnotes.example. During a major
incident, that page is updated at least every 30 minutes until the issue is
resolved.
