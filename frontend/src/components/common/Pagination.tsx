import type { PaginationMeta } from "../../types/api";

export function PaginationBar({
  pagination,
  onPageChange,
}: {
  pagination: PaginationMeta;
  onPageChange: (page: number) => void;
}) {
  if (pagination.total === 0) return null;

  const start = (pagination.page - 1) * pagination.page_size + 1;
  const end = Math.min(pagination.page * pagination.page_size, pagination.total);

  return (
    <div className="pagination-bar">
      <span>
        Showing {start}–{end} of {pagination.total}
      </span>
      <div className="pagination-controls">
        <button
          type="button"
          className="btn btn-secondary"
          disabled={pagination.page <= 1}
          onClick={() => onPageChange(pagination.page - 1)}
        >
          Previous
        </button>
        <span>
          Page {pagination.page} of {Math.max(pagination.total_pages, 1)}
        </span>
        <button
          type="button"
          className="btn btn-secondary"
          disabled={pagination.page >= pagination.total_pages}
          onClick={() => onPageChange(pagination.page + 1)}
        >
          Next
        </button>
      </div>
    </div>
  );
}
