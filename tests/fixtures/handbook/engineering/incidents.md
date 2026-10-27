# Handling an incident

Anyone can declare an incident, and when in doubt you should declare one. Type `/incident` in any Slack channel to open a dedicated incident channel.

## Severity

A minor incident does not page anyone and can wait for working hours. A major incident pages the on-call engineer immediately. A critical incident means the product is down for customers.

## Useful commands

Restart a stuck worker with:

```bash
# this comment is not a heading
kubectl rollout restart deployment/worker
```

## After the incident

The incident lead writes a post-mortem within five working days and shares it in the engineering channel.
