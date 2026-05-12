import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import App from './App';

beforeEach(() => {
  URL.createObjectURL = jest.fn(() => 'blob:preview');
  URL.revokeObjectURL = jest.fn();

  global.fetch = jest.fn((url, options) => {
    if (!options) {
      return Promise.resolve({
        ok: true,
        json: async () => ({
          status: 'ok',
          model_loaded: true,
          classes: ['W180', 'W210', 'W300', 'W500'],
        }),
      });
    }

    return Promise.resolve({
      ok: true,
      json: async () => ({
        predicted_grade: 'W210',
        confidence: 0.82,
        class_probabilities: {
          W180: 0.1,
          W210: 0.82,
          W300: 0.06,
          W500: 0.02,
        },
      }),
    });
  });
});

afterEach(() => {
  jest.resetAllMocks();
});

test('renders the updated hero and backend status', async () => {
  render(<App />);

  expect(screen.getByText(/grade every cashew/i)).toBeInTheDocument();
  await waitFor(() => expect(screen.getByText(/backend online/i)).toBeInTheDocument());
});

test('runs prediction when a file is selected', async () => {
  render(<App />);

  const file = new File(['kernel'], 'cashew.jpg', { type: 'image/jpeg' });
  const input = document.querySelector('input[type="file"]');

  fireEvent.change(input, { target: { files: [file] } });
  fireEvent.click(screen.getByText(/start grading run/i));

  await waitFor(() => expect(screen.getByTestId('predicted-grade')).toHaveTextContent('W210'));
  expect(screen.getByTestId('business-band')).toHaveTextContent(/jumbo/i);
});
