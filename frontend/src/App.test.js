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
          classes: ['s0', 's1', 's2'],
        }),
      });
    }

    return Promise.resolve({
      ok: true,
      json: async () => ({
        predicted_grade: 's1',
        confidence: 0.82,
        class_probabilities: {
          s0: 0.1,
          s1: 0.82,
          s2: 0.08,
        },
      }),
    });
  });
});

afterEach(() => {
  jest.resetAllMocks();
});

test('renders dashboard heading and backend status', async () => {
  render(<App />);

  expect(screen.getByText(/cashew quality analytics and grading system/i)).toBeInTheDocument();
  await waitFor(() => expect(screen.getByText(/connected/i)).toBeInTheDocument());
});

test('runs prediction when a file is selected', async () => {
  render(<App />);

  const file = new File(['kernel'], 'cashew.jpg', { type: 'image/jpeg' });
  const input = document.querySelector('input[type="file"]');

  fireEvent.change(input, { target: { files: [file] } });
  fireEvent.click(screen.getByText(/start grading run/i));

  await waitFor(() => expect(screen.getByTestId('predicted-grade')).toHaveTextContent('s1'));
  expect(screen.getByTestId('business-band')).toHaveTextContent(/premium range/i);
});
