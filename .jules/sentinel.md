## 2024-05-22 - [BentoML Service Input Limits]
**Vulnerability:** The BentoML service lacked input validation for batch size, image dimensions, and query length. This could allow Denial of Service (DoS) attacks by sending massive payloads (e.g., 100k images in a batch, 10k x 10k images, or megabyte-sized queries) that would exhaust server resources. Additionally, exception handling was returning raw error strings, potentially leaking internal implementation details.
**Learning:** BentoML's `@api` decorators provide structure but do not automatically enforce resource limits or sanitize inputs. Explicit validation logic must be added inside the service methods.
**Prevention:**
1.  Define hard limits for all inputs (`MAX_BATCH_SIZE`, `MAX_IMAGE_SIZE`, `MAX_QUERY_LENGTH`).
2.  Validate inputs at the start of every API method and return specific, user-friendly error messages if limits are exceeded.
3.  Catch generic exceptions, log the full traceback internally, but return a generic "Internal Server Error" message to the client to prevent info leakage.
