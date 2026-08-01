# Practice policy hosting

`practice-policy.yaml` creates a retained DynamoDB table in `us-west-2` with fixed 25 RCU / 25 WCU provisioned capacity. It has no autoscaling, streams, GSIs, or paid backups. It also creates capacity alarms, an AWS Budget with email notifications, an MFA-gated importer role, and (only when parameters are supplied) a Vercel OIDC read role.

The browser never receives AWS credentials. Next.js reads shards server-side.
Locally, it uses the normal short-lived AWS CLI/SSO credential chain. On
Vercel, `@vercel/oidc-aws-credentials-provider` exchanges the workload token
for the template's read-only role; configure:

```text
PRACTICE_POLICY_TABLE=poker-lab-practice-policy
AWS_REGION=us-west-2
AWS_ROLE_ARN=<VercelPolicyReadRoleArn output>
```

If the identity provider uses a custom audience, also set
`VERCEL_OIDC_AUDIENCE` to the same value as `VercelOidcAudience`. Otherwise the
provider uses Vercel's team audience from the original workload token.

No permanent AWS access key belongs in `.env`, Vercel, the browser, or the repository.

## One-time deployment inputs

- AWS account plus a one-time AWS CLI/SSO authorization (no root password or root key)
- Billing alert email
- ARN of the non-root IAM/IAM Identity Center role that may assume the importer role
- Later, the global-issuer Vercel OIDC provider ARN, team audience (normally
  `https://vercel.com/TEAM_SLUG`), and exact/wildcard team-project subject

The template currently targets Vercel's **Global** issuer
(`https://oidc.vercel.com`). Team-issuer provider ARNs include the team slug in
the IAM condition-key prefix and should use a separately generated trust
policy once the team identifier is known.

Confirm the AWS Budget email subscription after deploying. Keep the stack in `us-west-2`.

## Immutable import flow

1. Produce a policy export JSON with one accepted manifest,
   SHA-256-addressed policy nodes, and any reachable postflop samples/replay
   histories. Policy shards use `PLP1`; sample shards use `PLS1`.
2. `npm run policy:export -- --input policy.json --output artifacts/hosted --existing-hosted-bytes 0`
3. Review `export-index.json` and run `npm run policy:size -- --indexes <index>`.
4. Assume the importer role through SSO/MFA.
5. `npm run policy:import -- --index <index> --table <table>` (resumable and throttled to 25 WCU).
6. `npm run policy:verify -- --index <index> --table <table>`.
7. Activate only the accepted manifest with `npm run policy:activate -- --table <table> --manifest <manifest.json>`.

Rollback uses the immutable activation history:

```text
npm run policy:activate -- --table <table> --rollback-version <previous-version>
```

The exporter refuses failed activation gates and projected hosted storage above
20GB. Its audit includes a conservative per-item DynamoDB metadata allowance,
not just compact payload bytes. DynamoDB items are split to 24KB payloads and
remain below the 400KB service limit.
